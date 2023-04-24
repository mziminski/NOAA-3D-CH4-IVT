# Used resources: https://stackoverflow.com/questions/58844463/how-to-get-a-list-of-every-point-inside-a-multipolygon-using-shapely 

from functools import singledispatch
from itertools import chain

from typing import (List, 
                    Tuple,
                    TypeVar)

from shapely.geometry import (GeometryCollection,
                              LinearRing,
                              LineString,
                              Point,
                              Polygon)

from shapely.geometry.base import (BaseGeometry,
                                   BaseMultipartGeometry)

Geometry = TypeVar('Geometry', bound=BaseGeometry)


@singledispatch
def to_coords(geometry: Geometry) -> List[Tuple[float, float]]:
    """Returns a list of unique vertices of a given geometry object."""
    raise NotImplementedError(f"Unsupported Geometry {type(geometry)}")


@to_coords.register
def _(geometry: Point):
    return [(geometry.x, geometry.y)]


@to_coords.register
def _(geometry: LineString):
    return list(geometry.coords)


# @to_coords.register
# def _(geometry: LinearRing):
#     return list(geometry.coords[:]) # removeing -1 from [:,-1] fixed lines not connecting in closed polygons


@to_coords.register
def _(geometry: BaseMultipartGeometry):
    geometry = geometry.geoms
    return list(chain.from_iterable(map(to_coords, geometry)))  + [None] # added [None] to separate multipart geometries, removed set() from list(set(...))


@to_coords.register
def _(geometry: Polygon):
    return to_coords(GeometryCollection([geometry.exterior, *geometry.interiors]))