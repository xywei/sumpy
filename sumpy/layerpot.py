from __future__ import division

import numpy as np
import loopy as lp
from pytools import memoize_method

from sumpy.tools import KernelComputation




def pop_expand(kernel, order, avec, bvec):
    dimensions = len(avec)
    from pytools import (
            generate_nonnegative_integer_tuples_summing_to_at_most
            as gnitstam)

    multi_indices = sorted(gnitstam(order, dimensions), key=sum)

    from sumpy.tools import mi_factorial, mi_power, mi_derivative
    return sum(
            mi_power(bvec, mi)/mi_factorial(mi) 
            * (-1)**sum(mi) # we're expanding K(-a)
            * mi_derivative(kernel, avec, mi)
            for mi in multi_indices)




class LayerPotential(KernelComputation):
    def __init__(self, ctx, kernels, order, density_usage=None,
            value_dtypes=None, strength_dtypes=None,
            geometry_dtype=None,
            options=[], name="layerpot", device=None):
        """
        :arg kernels: list of :class:`sumpy.kernel.Kernel` instances
        :arg density_usage: A list of integers indicating which expression
          uses which density. This implicitly specifies the
          number of density arrays that need to be passed.
          Default: all kernels use the same density.
        """
        KernelComputation.__init__(self, ctx, kernels, density_usage,
                value_dtypes, strength_dtypes, geometry_dtype,
                name, options, device)

        from pytools import single_valued
        self.dimensions = single_valued(knl.dimensions for knl in self.kernels)

        self.order = order

    @memoize_method
    def get_kernel(self):
        from sumpy.symbolic import make_sym_vector

        avec = make_sym_vector("a", self.dimensions)
        bvec = make_sym_vector("b", self.dimensions)

        from sumpy.codegen import sympy_to_pymbolic_for_code
        exprs = sympy_to_pymbolic_for_code(
                [pop_expand(k.get_expression(avec), self.order, avec, bvec)
                    for i, k in enumerate(self.kernels)])
        from pymbolic import var
        exprs = [var("density_%d" % i)[var("isrc")]*expr
                for i, expr in enumerate(exprs)]

        geo_dtype = self.geometry_dtype
        arguments = (
                [
                   lp.ArrayArg("src", geo_dtype, shape=("nsrc", self.dimensions), order="C"),
                   lp.ArrayArg("tgt", geo_dtype, shape=("ntgt", self.dimensions), order="C"),
                   lp.ArrayArg("center", geo_dtype, shape=("ntgt", self.dimensions), order="C"),
                   lp.ScalarArg("nsrc", np.int32),
                   lp.ScalarArg("ntgt", np.int32),
                ]+[
                   lp.ArrayArg("density_%d" % i, dtype, shape="nsrc", order="C")
                   for i, dtype in enumerate(self.strength_dtypes)
                ]+[
                   lp.ArrayArg("result_%d" % i, dtype, shape="ntgt", order="C")
                   for i, dtype in enumerate(self.value_dtypes)
                   ]
                + self.gather_arguments())

        from pymbolic import parse
        knl = lp.make_kernel(self.device,
                "[nsrc,ntgt] -> {[isrc,itgt,idim]: 0<=itgt<ntgt and 0<=isrc<nsrc "
                "and 0<=idim<%d}" % self.dimensions,
                [
                "[|idim] <%s> a[idim] = center[itgt,idim] - src[isrc,idim]" % geo_dtype.name,
                "[|idim] <%s> b[idim] = tgt[itgt,idim] - center[itgt,idim]" % geo_dtype.name,
                ]+[
                lp.Instruction(id=None,
                    assignee=parse("pair_result_%d" % i), expression=expr,
                    temp_var_type=dtype)
                for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))
                ]+[
                "result_%d[itgt] = sum_%s(isrc, pair_result_%d)" % (i, dtype.name, i)
                for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))],
                arguments,
                name="layerpot", assumptions="nsrc>=1 and ntgt>=1",
                preamble=self.gather_preambles())

        return knl

    def get_optimized_kernel(self):
        knl = self.get_kernel()
        knl = lp.split_dimension(knl, "itgt", 1024, outer_tag="g.0")
        return knl

    def __call__(self, queue, targets, sources, centers, densities, **kwargs):
        cknl = self.get_compiled_kernel()

        for i, dens in enumerate(densities):
            kwargs["density_%d" % i] = dens

        return cknl(queue, src=sources, tgt=targets, center=centers,
                nsrc=len(sources), ntgt=len(targets), **kwargs)