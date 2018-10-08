from math import sin
import unittest
import numpy as np

from pytest import importorskip

from ..config import get_config, use_config
from ..array import wrap
from ..jit import get_binop_return_type, AnnotationHelper
from ..types import annotate
from ..parallel import Elementwise, Reduction, Scan


@annotate
def g(x):
    return x


@annotate
def h(a, b):
    return g(a) * g(b)


class TestAnnotationHelper(unittest.TestCase):
    def test_const_as_call_arg(self):
        # Given
        @annotate
        def int_f(a):
            return g(1)

        # When
        types = {'a': 'int'}
        helper = AnnotationHelper(int_f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

        # Given
        @annotate
        def long_f(a):
            return g(10000000000)

        # When
        types = {'a': 'int'}
        helper = AnnotationHelper(long_f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'long'

        # Given
        @annotate
        def double_f(a):
            return g(1.)

        # When
        types = {'a': 'int'}
        helper = AnnotationHelper(double_f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'double'

    def test_variable_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            x = declare('int')
            x = a + b
            return g(x)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_subscript_as_call_arg(self):
        # Given
        @annotate
        def f(i, a):
            return g(a[i])

        # When
        types = {'i': 'int', 'a': 'intp'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_binop_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            return g(a + b)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_compare_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            return g(a == b)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_call_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            return g(h(a, b))

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_binop_with_call_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            return g(h(a, b) + h(b, a))

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_non_jit_call_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            return g(sin(a))

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'double'

    def test_variable_in_return(self):
        # Given
        @annotate
        def f(a):
            return a

        # When
        types = {'a': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

    def test_subscript_in_return(self):
        # Given
        @annotate
        def f(i, a):
            return a[i]

        # When
        types = {'i': 'int', 'a': 'intp'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

    def test_const_in_return(self):
        # Given
        @annotate
        def int_f(a, b):
            return 1

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(int_f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

        # Given
        @annotate
        def long_f(a, b):
            return 10000000000

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(long_f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'long'

        # Given
        @annotate
        def double_f(a, b):
            return 1.

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(double_f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'double'

    def test_binop_in_return(self):
        # Given
        @annotate
        def f(a, b):
            return a + b

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

    def test_call_in_return(self):
        # Given
        @annotate
        def f(a, b):
            return g(a)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert 'g' in helper.external_funcs
        assert helper.arg_types['return_'] == 'int'

    def test_binop_with_call_in_return(self):
        # Given
        @annotate
        def f(a, b):
            return g(a) + g(b)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

    def test_multi_level_call_in_return(self):
        # Given
        @annotate
        def f(a, b):
            return h(a, b)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert 'h' in helper.external_funcs
        assert 'g' in helper.external_funcs['h'].external_funcs
        assert helper.arg_types['return_'] == 'int'

    def test_non_jit_call_in_return(self):
        # Given
        @annotate
        def f(a):
            return sin(a)

        # When
        types = {'a': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'double'

    def test_binop_return_type(self):
        # Given
        @annotate
        def f(a, b):
            return a + b

        # When
        types = {'a': 'long', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'long'

        # When
        types = {'a': 'int', 'b': 'double'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'double'

        # When
        types = {'a': 'uint', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

        # When
        types = {'a': 'uint', 'b': 'ulong'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'ulong'


class TestParallelJIT(unittest.TestCase):
    def setUp(self):
        cfg = get_config()
        self._use_double = cfg.use_double
        cfg.use_double = True

    def tearDown(self):
        get_config().use_double = self._use_double

    def _check_simple_elementwise_jit(self, backend):
        # Given
        @annotate
        def axpb(i, x, y, a, b):
            y[i] = a * sin(x[i]) + b

        x = np.linspace(0, 1, 10000)
        y = np.zeros_like(x)
        a = 2.0
        b = 3.0
        x, y = wrap(x, y, backend=backend)

        # When
        e = Elementwise(axpb, backend=backend)
        e(x, y, a, b)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, a * np.sin(x.data) + b))

    def test_elementwise_jit_works_with_cython(self):
        self._check_simple_elementwise_jit(backend='cython')

    def test_elementwise_jit_works_with_opencl(self):
        importorskip('pyopencl')

        self._check_simple_elementwise_jit(backend='opencl')

    def test_elementwise_jit_works_with_cuda(self):
        importorskip('pycuda')

        self._check_simple_elementwise_jit(backend='cuda')

    def _check_simple_reduction_jit(self, backend):
        x = np.linspace(0, 1, 1000) / 1000
        x = wrap(x, backend=backend)

        # When
        r = Reduction('a+b', backend=backend)
        result = r(x)

        # Then
        self.assertAlmostEqual(result, 0.5, 6)

    def _check_reduction_min_jit(self, backend):
        x = np.linspace(0, 1, 1000) / 1000
        x = wrap(x, backend=backend)

        # When
        r = Reduction('min(a, b)', neutral='INFINITY', backend=backend)
        result = r(x)

        # Then
        self.assertAlmostEqual(result, 0.0, 6)

    def _check_reduction_with_map_jit(self, backend):
        # Given
        from math import cos, sin
        x = np.linspace(0, 1, 1000) / 1000
        y = x.copy()
        x, y = wrap(x, y, backend=backend)

        @annotate
        def map(i=0, x=[0.0], y=[0.0]):
            return cos(x[i]) * sin(y[i])

        # When
        r = Reduction('a+b', map_func=map, backend=backend)
        result = r(x, y)

        # Then
        self.assertAlmostEqual(result, 0.5, 6)

    def test_reduction_jit_works_without_map_cython(self):
        self._check_simple_reduction_jit(backend='cython')

    def test_reduction_jit_works_with_map_cython(self):
        self._check_reduction_with_map_jit(backend='cython')

    def test_reduction_jit_works_neutral_cython(self):
        self._check_reduction_min_jit(backend='cython')

    def test_reduction_jit_works_without_map_opencl(self):
        importorskip('pyopencl')
        self._check_simple_reduction_jit(backend='opencl')

    def test_reduction_jit_works_with_map_opencl(self):
        importorskip('pyopencl')
        self._check_reduction_with_map_jit(backend='opencl')

    def test_reduction_jit_works_neutral_opencl(self):
        importorskip('pyopencl')
        self._check_reduction_min_jit(backend='opencl')

    def test_reduction_jit_works_without_map_cuda(self):
        importorskip('pycuda')
        self._check_simple_reduction_jit(backend='cuda')

    def test_reduction_jit_works_with_map_cuda(self):
        importorskip('pycuda')
        self._check_reduction_with_map_jit(backend='cuda')

    def test_reduction_jit_works_neutral_cuda(self):
        importorskip('pycuda')
        self._check_reduction_min_jit(backend='cuda')

    def _test_scan_jit(self, backend):
        # Given
        a = np.arange(10000, dtype=np.int32)
        data = a.copy()
        expect = np.cumsum(data)

        a = wrap(a, backend=backend)

        @annotate
        def input_f(i, ary):
            return ary[i]

        @annotate
        def output_f(i, item, ary):
            ary[i] = item

        # When
        scan = Scan(input_f, output_f, 'a+b', dtype=np.int32,
                    backend=backend)
        scan(ary=a)

        a.pull()
        result = a.data

        # Then
        np.testing.assert_equal(expect, result)

    def test_scan_jit_works_cython(self):
        self._test_scan_jit(backend='cython')

    def test_scan_jit_works_cython_parallel(self):
        with use_config(use_openmp=True):
            self._test_scan_jit(backend='cython')

    def test_scan_jit_works_opencl(self):
        importorskip('pyopencl')
        self._test_scan_jit(backend='opencl')

    def test_scan_jit_works_cuda(self):
        importorskip('pycuda')
        self._test_scan_jit(backend='cuda')

    def _test_unique_scan_jit(self, backend):
        # Given
        a = np.random.randint(0, 100, 100, dtype=np.int32)
        a = np.sort(a)
        data = a.copy()

        unique_ary_actual = np.sort(np.unique(data))
        unique_count_actual = len(np.unique(data))

        a = wrap(a, backend=backend)

        unique_ary = np.zeros(len(a.data), dtype=np.int32)
        unique_ary = wrap(unique_ary, backend=backend)

        unique_count = np.zeros(1, dtype=np.int32)
        unique_count = wrap(unique_count, backend=backend)

        @annotate
        def input_f(i, ary):
            if i == 0 or ary[i] != ary[i - 1]:
                return 1
            else:
                return 0

        @annotate
        def output_f(i, prev_item, item, N, ary, unique, unique_count):
            if item != prev_item:
                unique[item - 1] = ary[i]
            if i == N - 1:
                unique_count[0] = item

        # When
        scan = Scan(input_f, output_f, 'a+b', dtype=np.int32, backend=backend)
        scan(ary=a, unique=unique_ary, unique_count=unique_count)
        unique_ary.pull()
        unique_count.pull()
        unique_count = unique_count.data[0]

        # Then
        self.assertTrue(unique_count == unique_count_actual)
        np.testing.assert_equal(unique_ary_actual,
                                unique_ary.data[:unique_count])

    def test_unique_scan_jit_cython(self):
        self._test_unique_scan_jit(backend='cython')

    def test_unique_scan_jit_cython_parallel(self):
        with use_config(use_openmp=True):
            self._test_unique_scan_jit(backend='cython')

    def test_unique_scan_jit_opencl(self):
        importorskip('pyopencl')
        self._test_unique_scan_jit(backend='opencl')

    def test_unique_scan_jit_cuda(self):
        importorskip('pycuda')
        self._test_unique_scan_jit(backend='cuda')

    def _get_segmented_scan_actual(self, a, segment_flags):
        output_actual = np.zeros_like(a)
        for i in range(len(a)):
            if segment_flags[i] == 0 and i != 0:
                output_actual[i] = output_actual[i - 1] + a[i]
            else:
                output_actual[i] = a[i]
        return output_actual

    def _test_segmented_scan_jit(self, backend):
        # Given
        a = np.random.randint(0, 100, 50000, dtype=np.int32)
        a_copy = a.copy()

        seg = np.random.randint(0, 100, 50000, dtype=np.int32)
        seg = (seg == 0).astype(np.int32)
        seg_copy = seg.copy()

        a = wrap(a, backend=backend)
        seg = wrap(seg, backend=backend)

        @annotate
        def input_f(i, ary):
            return ary[i]

        @annotate
        def segment_f(i, seg_flag):
            return seg_flag[i]

        @annotate
        def output_f(i, item, ary):
            ary[i] = item

        output_actual = self._get_segmented_scan_actual(a_copy, seg_copy)

        # When
        scan = Scan(input_f, output_f, 'a+b', dtype=np.int32,
                    backend=backend, is_segment=segment_f)
        scan(ary=a, seg_flag=seg)
        a.pull()

        # Then
        np.testing.assert_equal(output_actual, a.data)

    def test_segmented_scan_jit_cython(self):
        self._test_segmented_scan_jit(backend='cython')

    def test_segmented_scan_jit_cython_parallel(self):
        with use_config(use_openmp=True):
            self._test_segmented_scan_jit(backend='cython')

    def test_segmented_scan_jit_opencl(self):
        importorskip('pyopencl')
        self._test_segmented_scan_jit(backend='opencl')

    def test_segmented_scan_jit_cuda(self):
        importorskip('pycuda')
        self._test_segmented_scan_jit(backend='cuda')

    def _test_scan_jit_last_item(self, backend):
        # Given
        a = np.random.randint(0, 100, 50000, dtype=np.int32)
        a_copy = a.copy()

        a = wrap(a, backend=backend)

        @annotate
        def output_f(i, last_item, item, ary):
            ary[i] = item + last_item

        expect = np.cumsum(a_copy) + np.cumsum(a_copy)[-1]

        # When
        scan = Scan(output=output_f, scan_expr='a+b',
                    dtype=np.int32, backend=backend)
        scan(input=a, ary=a)
        a.pull()

        # Then
        np.testing.assert_equal(expect, a.data)

    def test_scan_jit_last_item_cython_parallel(self):
        with use_config(use_openmp=True):
            self._test_scan_jit_last_item(backend='cython')

    def test_scan_jit_last_item_opencl(self):
        importorskip('pyopencl')
        self._test_scan_jit_last_item(backend='opencl')

    def test_scan_jit_last_item_cuda(self):
        importorskip('pycuda')
        self._test_scan_jit_last_item(backend='cuda')
