import sys

sys.path.append("../util")


def test_gen_uocp_poly():
    from generateOCP_poly_mon import gen_uocp_poly

    gen_uocp_poly()


def test_gen_uocp():
    from generateOCP import gen_uocp

    gen_uocp()


if __name__ == "__main__":
    test_gen_uocp_poly()
    test_gen_uocp()
    pass
