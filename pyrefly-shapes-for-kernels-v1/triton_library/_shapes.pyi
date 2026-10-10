"""Shape expressions for logical offset grids and their address steps."""

from shape_extensions import Int, IntTuple, dsl, type_shape_dsl_function

@type_shape_dsl_function
def insert_extent(tile: IntTuple, axis: int) -> IntTuple:
    if axis < 0 or axis > len(tile):
        return dsl.Invalid("offset axis out of range")
    return dsl.concat(tile[:axis], dsl.concat(dsl.IntTuple((1,)), tile[axis:]))

@type_shape_dsl_function
def insert_step(tile: IntTuple, steps: IntTuple, axis: int) -> IntTuple:
    if len(tile) != len(steps):
        return dsl.Invalid("offset extents and address steps must have equal rank")
    if axis < 0 or axis > len(steps):
        return dsl.Invalid("offset axis out of range")
    return dsl.concat(steps[:axis], dsl.concat(dsl.IntTuple((0,)), steps[axis:]))

@type_shape_dsl_function
def scale_steps(tile: IntTuple, steps: IntTuple, scale: Int) -> IntTuple:
    if len(tile) != len(steps):
        return dsl.Invalid("offset extents and address steps must have equal rank")
    # A generator retains each symbolic step and its position for known-rank tuples.
    return dsl.IntTuple((step * scale for step in steps))
