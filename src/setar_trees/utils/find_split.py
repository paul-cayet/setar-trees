import numpy as np
from numpy.linalg import solve, LinAlgError


def find_cut_point(X, y, x_ix, k, lag, criterion="RSS"):
    """
    TODO:
    - Change name k
    - Change x_ik
    - Remove lag if not used
    - ^So refactor a lot of things
    - Change code so that it can run with nopython
    - Add documentation
    - Need to check that matrix X does is not singular /!\
    -> no more cols than rows and no linear combi
    """
    eps = 1e-8  # min RSS, to avoid errors in log computation with AIC
    recheck = 0
    n_rows, n_cols = X.shape

    # Split into chunks
    q = np.linspace(
        x_ix.min(), x_ix.max(), k
    )  # s in the paper (candidate thresholds)
    k = len(q)
    q = np.concatenate(([-np.inf], q, [np.inf]))

    # Lists to store the left and right statistics for each interval
    XtX_list = [np.zeros((n_cols, n_cols))] * (k + 1)
    Xty_list = [np.zeros((n_cols, 1))] * (k + 1)
    yty_list = np.zeros(k + 1)
    n_s = np.zeros(k + 1)

    XtX_left = np.zeros((n_cols, n_cols))  # B(k) in the paper
    Xty_left = np.zeros((n_cols, 1))  # c(k) in the paper
    XtX_right = np.zeros((n_cols, n_cols))  # XtX - B(k)
    Xty_right = np.zeros((n_cols, 1))  # Xty - c(k)
    yty_left = 0  # d(k) in the paper
    yty_right = 0  # yty - d(k)

    b_left = np.zeros((n_cols, k))  # beta(Lk) in the paper
    b_right = np.zeros((n_cols, k))  # beta(Rk) in the paper

    RSS_left = np.zeros(k)  # SSE(Lk) in the paper
    RSS_right = np.zeros(k)  # SSE(Rk) in the paper
    AICc_left = np.zeros(k)
    AICc_right = np.zeros(k)

    n_left = 0
    n_right = n_rows

    # computing inner products
    for i in range(k + 1):
        ix = np.where((x_ix >= q[i]) & (x_ix < q[i + 1]))[0]

        # if len(ix) == 0 we don't do anything
        if len(ix) > 0:
            XtX_list[i] = np.matmul(X[ix].T, X[ix])
            Xty_list[i] = np.dot(X[ix].T, y[ix]).reshape((-1, 1))

            assert XtX_list[i].shape == (n_cols, n_cols), XtX_list[
                i
            ].shape  # TO REMOVE
            assert Xty_list[i].shape == (n_cols, 1), Xty_list[
                i
            ].shape  # TO REMOVE

            yty_list[i] = np.sum(np.square(y[ix]))

            XtX_right += XtX_list[i]
            Xty_right += Xty_list[i]
            yty_right += yty_list[i]

        n_s[i] = len(ix)
    # print(XtX_list)
    # print(n_s)

    # Do the partitioning
    for i in range(k):
        XtX_left += XtX_list[i]
        Xty_left += Xty_list[i]
        yty_left += yty_list[i]
        n_left += n_s[i]

        XtX_right -= XtX_list[i]
        Xty_right -= Xty_list[i]
        yty_right -= yty_list[i]
        n_right -= n_s[i]

        if n_left > 0 and n_right > 0:
            try:
                # print('XtX_left')
                # print([XtX_left])
                # print('Xty_left')
                # print([Xty_left])
                b_left[:, i] = solve(
                    XtX_left, Xty_left
                ).flatten()  # equivalent to the inverse notation in paper
                b_right[:, i] = solve(XtX_right, Xty_right).flatten()  # same

                # Compute the RSS
                RSS_left[i] = yty_left - np.dot(
                    b_left[:, i].T, np.dot(XtX_left, b_left[:, i])
                )  # I suspect there may be some stability issues here
                RSS_right[i] = yty_right - np.dot(
                    b_right[:, i].T, np.dot(XtX_right, b_right[:, i])
                )  # same

                # To account for stability issues (will need to be fixed later)
                RSS_left[i] = max(eps, RSS_left[i])
                RSS_right[i] = max(eps, RSS_right[i])

                # Compute the model scores
                AICc_left[i] = (
                    n_left * np.log(2 * np.pi * RSS_left[i] / n_left)
                    + n_left
                    + (n_cols + 1) * n_left / (n_left - n_cols - 1)
                )
                AICc_right[i] = (
                    n_right * np.log(2 * np.pi * RSS_right[i] / n_right)
                    + n_right
                    + (n_cols + 1) * n_right / (n_right - n_cols - 1)
                )
            except LinAlgError as e:
                recheck += 1
                b_left[:, i] = np.inf
                b_right[:, i] = np.inf
                RSS_left[i] = np.inf
                RSS_right[i] = np.inf
                AICc_left[i] = np.inf
                AICc_right[i] = np.inf
                raise e
        else:
            b_left[:, i] = np.inf
            b_right[:, i] = np.inf
            RSS_left[i] = np.inf
            RSS_right[i] = np.inf
            AICc_left[i] = np.inf
            AICc_right[i] = np.inf

    # Find the best split and return it
    cutpoint_penalty = np.log(k)

    AICc = AICc_left + AICc_right
    RSS = RSS_left + RSS_right

    if criterion == "AICc":
        I = np.argmin(AICc)  # noqa: E741
    else:
        I = np.argmin(RSS)  # noqa: E741

    # Create a "linear model" equivalent structure
    r_left = {
        "beta0": np.zeros(1),
        "beta": b_left[:, I],
        "RSS": RSS_left[I],
        "AICc": AICc_left[I],
    }

    r_right = {
        "beta0": np.zeros(1),
        "beta": b_right[:, I],
        "RSS": RSS_right[I],
        "AICc": AICc_right[I],
    }

    return {
        "I": I,
        "cut_point": q[I + 1],
        "b_left": b_left[:, I],
        "b_right": b_right[:, I],
        "RSS_left": RSS_left[I],
        "RSS_right": RSS_right[I],
        "AICc_left": AICc_left[I],
        "AICc_right": AICc_right[I],
        "RSS": RSS,
        "AICc": AICc,
        "cutpoint_penalty": cutpoint_penalty,
        "left_model": r_left,
        "right_model": r_right,
        "ix_left": x_ix <= q[I + 1],
        "q": q[1:-1],
        "need_recheck": recheck,
    }
