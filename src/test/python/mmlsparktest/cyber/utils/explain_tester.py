__author__ = 'rolevin'

from typing import Any, Callable, List

from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from mmlsparktest.spark import *


class ExplainTester:
    def check_explain(self, explainable: Any, params: List[str], type_count_checker: Callable):
        explained = explainable.explainParams()

        assert len(explained.split('\n')) == len(params), explained

        def to_camel_case(prefix: str, name: str) -> str:
            if name == 'inputCol' and isinstance(explained, HasInputCol):
                return prefix + 'InputCol'
            elif name == 'outputCol' and isinstance(explained, HasOutputCol):
                return prefix + 'OutputCol'
            else:
                parts = name.split('_')
                return prefix + ''.join([parts[i][0:1].upper() + parts[i][1:] for i in range(len(parts))])

        values = []

        for pp in params:
            assert pp in explained, explained
            getter_method_name = to_camel_case('get', pp)
            ret_value = getattr(type(explainable), getter_method_name)(explainable)
            values.append(ret_value)

            # test setter
            setter_method_name = to_camel_case('set', pp)
            getattr(type(explainable), setter_method_name)(explainable, ret_value)
            re_ret_value = getattr(type(explainable), getter_method_name)(explainable)

            # test that value stays the same
            assert re_ret_value == ret_value

        def count_instance(arr, tt):
            return len([vv for vv in arr if (tt is not None and isinstance(vv, tt)) or (tt is None and vv is None)])

        assert type_count_checker(count_instance(values, str), str)
        assert type_count_checker(count_instance(values, int), int)
        assert type_count_checker(count_instance(values, float), float)
        assert type_count_checker(count_instance(values, bool), bool)
        assert type_count_checker(count_instance(values, None), None)
