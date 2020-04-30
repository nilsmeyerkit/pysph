"""Convenience script for running various PySPH related tasks.
"""

from __future__ import print_function

from argparse import ArgumentParser
from os.path import exists, join
import sys


def run_viewer(args):
    from pysph.tools.mayavi_viewer import main
    main(args)


def run_examples(args):
    from pysph.examples.run import main
    main(args)


def output_vtk(args):
    from pysph.solver.vtk_output import main
    main(args)


def _has_pysph_dir():
    init_py = join('pysph', '__init__.py')
    init_pyc = join('pysph', '__init__.pyc')
    return exists(init_py) or exists(init_pyc)


def run_tests(args):
    argv = ['--pyargs', 'pysph'] + args
    from pytest import cmdline
    cmdline.main(args=argv)


def make_binder(args):
    from pysph.tools.binder import main
    main(args)


def cull_files(args):
    from pysph.tools.cull import main
    main(args)


def main():
    parser = ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "-h", "--help", action="store_true", default=False, dest="help",
        help="show this help message and exit"
    )
    subparsers = parser.add_subparsers(help='sub-command help')

    viewer = subparsers.add_parser(
        'view', help='View output files generated by PySPH',
        add_help=False
    )
    viewer.set_defaults(func=run_viewer)

    runner = subparsers.add_parser(
        'run', help='Run PySPH examples',
        add_help=False
    )
    runner.set_defaults(func=run_examples)

    vtk_out = subparsers.add_parser(
        'dump_vtk', help='Dump VTK Output',
        add_help=False
    )
    vtk_out.set_defaults(func=output_vtk)

    tests = subparsers.add_parser(
        'test', help='Run entire PySPH test-suite',
        add_help=False
    )
    tests.set_defaults(func=run_tests)

    binder = subparsers.add_parser(
        'binder',
        help='Make a mybinder.org compatible directory for upload to a ' +
             'host repo',
        add_help=False
    )
    binder.set_defaults(func=make_binder)

    cull = subparsers.add_parser(
        'cull',
        help='Cull files in a given directory by a specified culling_factor',
        add_help=False
    )
    cull.set_defaults(func=cull_files)

    if (len(sys.argv) == 1 or (len(sys.argv) > 1 and
                               sys.argv[1] in ['-h', '--help'])):
        parser.print_help()
        sys.exit()

    args, extra = parser.parse_known_args()
    args.func(extra)


if __name__ == '__main__':
    main()
