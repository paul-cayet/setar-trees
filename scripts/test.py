import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--none",
        help="",
        default="",
        required=False,
    )  # noqa: E501

    args = parser.parse_args()

    print("Not implemented yet")
