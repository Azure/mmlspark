# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

from py4j.protocol import Py4JError

def generateTypeConverter(name, cache, typeConverter):
    """
    Type converter

    Args:
        name (str):
        cache:
        typeConverter:

    Returns:
        lambda: Function to convert the type
    """
    return lambda value: typeConverter(name, value, cache)

def complexTypeConverter(name, value, cache):
    """
    Type conversion for complex types

    Args:
        name:
        value:
        cache:

    Returns:
        _java_obj:
    """
    cache[name]=value
    try:
        if isinstance(value, list):
            java_value=[]
            for v in value:
                if hasattr(v, "_transfer_params_to_java"):
                    v._transfer_params_to_java()
                java_value.append(v._java_obj)
            return java_value
        if hasattr(value, "_transfer_params_to_java"):
            value._transfer_params_to_java()
        if hasattr(value, "_java_obj"):
            return value._java_obj
        else:
            return value._to_java()
    except Py4JError as e:
        return value
