from __future__ import annotations
from functools import partial
import sys
import copy
import inspect
import json
import os
import types
import typing
from typing import Any, Callable, Generator, Generic, TypeVar
import uuid
import importlib
import yaml
import random

class BaseUtil:
    @staticmethod
    def copy(obj: Any) -> Any:
        return obj.copy() if hasattr(obj, "copy") else copy.copy(obj)

    @staticmethod
    def deepcopy(obj: Any) -> Any:
        return obj.deepcopy() if hasattr(obj, "deepcopy") else copy.deepcopy(obj)

    @staticmethod
    def smart_copy(obj: Any, isDeep: bool=False) -> Any:
        if type(obj) in [list, tuple, set]:
            return type(obj)([BaseUtil.smart_copy(obj0, isDeep) for obj0 in obj])
        elif type(obj) is dict:
            return {BaseUtil.smart_copy(key, isDeep): BaseUtil.smart_copy(val, isDeep) for key, val in obj.items()}
        else:
            if isDeep:
                return BaseUtil.deepcopy(obj)
            else:
                return BaseUtil.copy(obj)

    @staticmethod
    def generate_key(obj: Any) -> tuple[type, ...]: # This might be slow.
        obj_key = [type(obj).__class__]
        for key, val in obj.__dict__.items():
            if hasattr(val, 'get_key') and callable(val.get_key):
                obj_key.append((key, val.get_key()))
            # elif key == '_objects' and type(val) is list:
            #     inner_list = [
            #         (_obj.get_key() if hasattr(_obj, "get_key") and callable(_obj.get_key) else BaseUtil.generate_key(_obj))
            #         for _obj in obj._objects
            #     ]
            #     obj_key.append((key, list, tuple(inner_list)))
            elif type(val) in [list, set]:
                # Assume that it isn't nested.
                obj_key.append((key, type(val), tuple(val)))
            elif type(val) is dict:
                obj_key.append((key, dict, tuple(val.keys()), tuple(val.values())))
            else:
                obj_key.append((key, val))
        
        return tuple(obj_key)
    
    @staticmethod
    def to_type_dict(val: Any | type | types.GenericAlias | typing._GenericAlias) -> dict[str, str]:
        if type(val) is type:
            typeDict = {
                '_module': val.__module__,
                '_qualname': val.__qualname__
            }
            if hasattr(val, '__orig_class__'):
                typeDict['_orig_class'] = BaseUtil.to_type_dict(getattr(val, '__orig_class__'))
            return typeDict
        elif type(val) in [types.GenericAlias, typing._GenericAlias]:
            return {
                '_module': val.__module__,
                '_qualname': val.__qualname__,
                '_args': [BaseUtil.to_type_dict(arg) for arg in val.__args__]
            }
        elif type(val) is TypeVar:
            return val.__name__
        else:
            if hasattr(val, '__orig_class__'):
                typeDict = BaseUtil.to_type_dict(val.__orig_class__)
                # typeDict['isOrigClass'] = True
            else:
                typeDict = BaseUtil.to_type_dict(type(val))
                # typeDict['isOrigClass'] = False
            return typeDict
    
    @staticmethod
    def from_type_dict(type_dict: dict[str, str]) -> type | types.GenericAlias | typing._GenericAlias:
        cls = getattr(
            importlib.import_module(type_dict['_module']),
            type_dict['_qualname']
        )
        if '_args' in type_dict:
            args = tuple([BaseUtil.from_type_dict(arg_dict) for arg_dict in type_dict['_args']])
            cls = cls[args]
        return cls

    @staticmethod
    def to_func_dict(func: Callable | partial) -> dict:
        assert callable(func)
        if type(func) is partial:
            keywordsDict: dict = dict()
            for key, val in func.keywords.items():
                if hasattr(val, 'to_dict'):
                    keywordsDict[key] = val.to_dict()
                elif callable(val): # Could be a callback function.
                    keywordsDict[key] = BaseUtil.to_func_dict(val)
                else:
                    keywordsDict[key] = val
            return dict(
                _funcModule=func.func.__module__,
                _funcQualname=func.func.__qualname__,
                _keywords=keywordsDict
            )
        else:
            return dict(
                _funcModule=func.__module__,
                _funcQualname=func.__qualname__
            )
    
    @staticmethod
    def from_func_dict(func_dict: dict, **kwargs) -> Callable | partial:
        module = importlib.import_module(func_dict['_funcModule'])
        attrSequence = func_dict['_funcQualname'].split('.')
        attr = getattr(module, attrSequence[0])
        for attr0 in attrSequence[1:]:
            attr = getattr(attr, attr0)
        assert callable(attr)
        func = attr
        if '_keywords' in func_dict:
            keywords: dict = {}
            for key, val in func_dict['_keywords'].items():
                if type(val) is dict and '_funcModule' in val:
                    keywords[key] = BaseUtil.from_func_dict(val, **kwargs)
                elif type(val) is dict:
                    assert '_typedict' in val
                    objCls = BaseUtil.from_type_dict(val['_typedict'])
                    if objCls is ContextVarRef:
                        assert 'context' in kwargs
                        obj = ContextVarRef.from_dict(val, context=kwargs['context'])
                    elif hasattr(objCls, 'from_dict'):
                        obj = objCls.from_dict(val)
                    else:
                        obj = objCls(**{key0: val0 for key0, val0 in val.items() if key0 != '_typedict'})
                    keywords[key] = obj
                else:
                    if type(val) is list:
                        val = tuple(val) # Seems like opencv's functions need tuples instead of lists.
                    keywords[key] = val
            return partial(func, **keywords)
        else:
            return func

class Base:
    """Base Class
    Assume that all class variables are in the parameter list of __init__
    """
    def __init__(self):
        pass

    def __str__(self) -> str:
        result = type(self).__name__
        result += "("
        param_str_list = [f"{key}={val}" for key, val in self.__dict__.items() if key != "__orig_class__"]
        result += ', '.join(param_str_list)
        result += ")"
        return result
    
    def __repr__(self) -> str:
        return self.__str__()

    def get_key(self):
        return json.dumps(self.to_dict(), sort_keys=True).encode()

    def __hash__(self):
        return hash(self.get_key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.get_key() == other.get_key()
        return NotImplemented

    @classmethod
    def get_constructor_params(cls) -> list[str]:
        return [param for param in list(inspect.signature(cls.__init__).parameters.keys()) if param != 'self']

    def to_dict(self, **kwargs) -> dict:
        if not hasattr(self, '__orig_class__'):
            item_dict = {
                key: (
                    val
                    if not hasattr(val, 'to_dict')
                    else val.to_dict()
                )
                for key, val in self.__dict__.items()
            }
            item_dict['_typedict'] = BaseUtil.to_type_dict(type(self))
        else:
            item_dict = {
                key: (
                    val
                    if not hasattr(val, 'to_dict')
                    else val.to_dict()
                )
                for key, val in self.__dict__.items()
                if key != '__orig_class__'
            }
            item_dict['_typedict'] = BaseUtil.to_type_dict(self.__orig_class__)
        return item_dict
    
    @classmethod
    def from_dict(cls, item_dict: dict, **kwargs):
        assert '_typedict' in item_dict
        loaded_cls = BaseUtil.from_type_dict(item_dict['_typedict'])
        for attr in ['__module__', '__qualname__']:
            assert getattr(cls, attr) == getattr(loaded_cls, attr), f"{cls.__name__}.{attr} != {loaded_cls.__name__}.{attr}"
        constructor_params = cls.get_constructor_params()
        constructor_dict = {}
        post_construction_dict = {}
        for key, val in item_dict.items():
            if key == '_typedict':
                continue
            if type(val) is dict and '_typedict' in val:
                inner_cls = BaseUtil.from_type_dict(val['_typedict'])
                assert hasattr(inner_cls, 'from_dict')
                inner = inner_cls.from_dict(val)
                if key in constructor_params:
                    constructor_dict[key] = inner
                else:
                    post_construction_dict[key] = inner
            else:
                if key in constructor_params:
                    constructor_dict[key] = val
                else:
                    post_construction_dict[key] = val
        obj = cls(**constructor_dict)
        if '_typedict' in item_dict and '_args' in item_dict['_typedict'] and cls.__module__ != 'builtins':
            obj.__dict__['__orig_class__'] = typing._GenericAlias(
                cls,
                tuple([BaseUtil.from_type_dict(arg_dict) for arg_dict in item_dict['_typedict']['_args']])
            )
        for key, val in post_construction_dict.items():
            assert hasattr(obj, key), f"Object of type {type(obj).__name__} has no attribute '{key}'"
            setattr(obj, key, val)
        return obj

    def save(self, path: str, **kwargs):
        ext = os.path.splitext(path)[1]
        if ext == '.json':
            json.dump(self.to_dict(**kwargs), open(path, 'w'), ensure_ascii=False, sort_keys=False)
        elif ext == '.yaml':
            yaml.dump(self.to_dict(**kwargs), open(path, 'w'), allow_unicode=True, sort_keys=False)
        else:
            raise Exception(f"Invalid file extension: {ext}")
    
    @classmethod
    def load(cls, path: str, **kwargs):
        ext = os.path.splitext(path)[1]
        if ext == '.json':
            return cls.from_dict(json.load(open(path, 'r')), **kwargs)
        elif ext == '.yaml':
            return cls.from_dict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader), **kwargs)
        else:
            raise Exception(f"Invalid file extension: {ext}")

    def _copy(self, isDeep: bool=False):
        params = self.get_constructor_params()
        constructor_dict = {}
        post_constructor_dict = {}
        for key, val in self.__dict__.items():
            val0 = BaseUtil.smart_copy(val, isDeep=isDeep) # might be slow
            if key in params:
                constructor_dict[key] = val0
            else:
                post_constructor_dict[key] = val0
        result = type(self)(**constructor_dict)
        for key, val in post_constructor_dict.items():
            setattr(result, key, val)
        return result

    def copy(self):
        """Shallow copy. Keep references."""
        return self._copy(isDeep=False)

    def deepcopy(self):
        """Deep copy. Copy to new location in memory."""
        return self._copy(isDeep=True)

T = TypeVar('T', bound=Base) # T can only be Base or a subtype of Base

class BaseHandler(Generic[T]):
    def __init__(self, _objects: list[T]=None):
        self._objects = _objects if _objects is not None else []
    
    def __str__(self) -> str:
        return f"[{', '.join([obj.__str__() for obj in self])}]"
    
    def __repr__(self) -> str:
        return self.__str__()

    def get_key(self) -> tuple:
        # obj_key = [type(self).__class__]
        # for key, val in self.__dict__.items():
        #     if key == '_objects':
        #         inner_list = [obj.get_key() for obj in self._objects]
        #         obj_key.append((key, list, tuple(inner_list)))
        #     elif hasattr(val, 'get_key') and callable(val.get_key):
        #         obj_key.append((key, val.get_key()))
        #     else:
        #         obj_key.append((key, val))
        # return tuple(obj_key)
        # # return BaseUtil.generate_key(self)
        return json.dumps(self.to_dict(), sort_keys=True).encode()

    def __hash__(self):
        return hash(self.get_key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.get_key() == other.get_key()
        return NotImplemented

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Generator[T]:
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, idx):
        if type(idx) is int:
            return self._objects[idx]
        elif type(idx) is slice:
            params = self.get_constructor_params()
            result = type(self)(**{key: val for key, val in self.__dict__.items() if key in params})
            result._objects = result._objects[idx.start:idx.stop:idx.step]
            for key, val in self.__dict__.items():
                if key not in params:
                    setattr(result, key, val)
            return result
        else:
            raise TypeError
    
    def __setitem__(self, idx, value):
        if type(idx) is int:
            self._objects[idx] = value
        elif type(idx) is slice:
            self._objects[idx.start:idx.stop:idx.step] = value
        else:
            raise TypeError

    def __delitem__(self, idx):
        if type(idx) is int:
            del self._objects[idx]
        elif type(idx) is slice:
            del self._objects[idx.start:idx.stop:idx.step]
        else:
            raise TypeError

    def _copy(self, isDeep: bool=False):
        params = self.get_constructor_params()
        constructor_dict = {}
        post_constructor_dict = {}
        for key, val in self.__dict__.items():
            val0 = BaseUtil.smart_copy(val, isDeep=isDeep) # might be slow
            if key in params:
                constructor_dict[key] = val0
            else:
                post_constructor_dict[key] = val0
        result = type(self)(**constructor_dict)
        for key, val in post_constructor_dict.items():
            setattr(result, key, val)
        return result

    def copy(self):
        """Shallow copy. Keep references."""
        return self._copy(isDeep=False)

    def deepcopy(self):
        """Deep copy. Copy to new location in memory."""
        return self._copy(isDeep=True)

    def get(self, func: Callable[[T], bool]=None, **kwargs) -> T | None:
        if func is not None:
            for obj in self:
                if func(obj):
                    return obj
        else:
            for obj in self:
                is_match = True
                for key, val in kwargs.items():
                    if getattr(obj, key) != val:
                        is_match = False
                        break
                if is_match:
                    return obj
        return None
    
    def search(self, func: Callable[[T], bool]=None, **kwargs):
        objects: list[T] = []
        if func is not None:
            for i, obj in enumerate(self):
                if func(obj):
                    objects.append(obj)
        else:
            for obj in self:
                is_match = True
                for key, val in kwargs.items():
                    if getattr(obj, key) != val:
                        is_match = False
                        break
                if is_match:
                    objects.append(obj)
        return type(self)(objects)
    
    @classmethod
    def get_constructor_params(cls) -> list[str]:
        return [param for param in list(inspect.signature(cls.__init__).parameters.keys()) if param != 'self']

    def _to_dict_compressed(self, **kwargs) -> dict:
        _typedict: dict = BaseUtil.to_type_dict(self)
        obj_typedict: dict = BaseUtil.to_type_dict(self[0]) if len(self) > 0 else None
        if '_args' in _typedict:
            del _typedict['_args']

        item_dict: dict = dict(_typedict=_typedict, obj_typedict=obj_typedict)
        for key, val in self.__dict__.items():
            if key == '__orig_class__':
                continue
            if key == '_objects':
                workingDictList: list[dict] = []
                for val0 in val:
                    workingDict = val0.to_dict(**kwargs)
                    assert '_typedict' in workingDict
                    del workingDict['_typedict']
                    workingDictList.append(workingDict)
                item_dict[key] = workingDictList
            elif hasattr(val, 'to_dict'):
                item_dict[key] = val.to_dict()
            else:
                item_dict[key] = val
        return item_dict

    def _to_dict_expanded(self, **kwargs) -> dict:
        _typedict: dict = BaseUtil.to_type_dict(self)
        # obj_typedict: dict = BaseUtil.to_type_dict(self[0]) if len(self) > 0 else None
        if '_args' in _typedict:
            del _typedict['_args']

        item_dict: dict = dict(_typedict=_typedict)
        for key, val in self.__dict__.items():
            if key == '__orig_class__':
                continue
            if key == '_objects':
                workingDictList: list[dict] = []
                for val0 in val:
                    workingDict = val0.to_dict(**kwargs)
                    assert '_typedict' in workingDict
                    # del workingDict['_typedict']
                    workingDictList.append(workingDict)
                item_dict[key] = workingDictList
            elif hasattr(val, 'to_dict'):
                item_dict[key] = val.to_dict()
            else:
                item_dict[key] = val
        return item_dict

    def to_dict(self, compressed: bool=True, **kwargs) -> dict:
        if compressed:
            return self._to_dict_compressed(**kwargs)
        else:
            return self._to_dict_expanded(**kwargs)

    @classmethod
    def _from_dict_compressed(cls, item_dict: dict, **kwargs):
        assert '_typedict' in item_dict
        assert 'obj_typedict' in item_dict
        _typedict = item_dict['_typedict']
        obj_typedict = item_dict['obj_typedict']
        cls_type = BaseUtil.from_type_dict(_typedict)
        obj_type = BaseUtil.from_type_dict(obj_typedict)

        constructor_params = cls.get_constructor_params()
        constructor_dict = {}
        post_construction_dict = {}
        for key, val in item_dict.items():
            if key in ['_typedict', 'obj_typedict']:
                continue
            if key == '_objects':
                assert type(val) is list
                if len(val) == 0:
                    constructor_dict[key] = val
                else:
                    loaded_val: list = []
                    for val0 in val:
                        val0['_typedict'] = obj_typedict
                        loaded_val.append(obj_type.from_dict(val0, **kwargs))
                    constructor_dict[key] = loaded_val
            elif type(val) is dict and '_typedict' in val:
                inner_cls = BaseUtil.from_type_dict(val['_typedict'])
                assert hasattr(inner_cls, 'from_dict')
                inner = inner_cls.from_dict(val)
                if key in constructor_params:
                    constructor_dict[key] = inner
                else:
                    post_construction_dict[key] = inner
            else:
                if key in constructor_params:
                    constructor_dict[key] = val
                else:
                    post_construction_dict[key] = val
        obj = cls(**constructor_dict)
        
        if '_typedict' in item_dict and '_args' in item_dict['_typedict'] and cls.__module__ != 'builtins':
            obj.__dict__['__orig_class__'] = typing._GenericAlias(
                cls,
                # tuple([BaseUtil.from_type_dict(arg_dict) for arg_dict in item_dict['_typedict']['_args']])
                (obj_type,)
            )

        for key, val in post_construction_dict.items():
            assert hasattr(obj, key), f"{type(obj).__name__} has no attribute {key}"
            setattr(obj, key, val)
        return obj

    @classmethod
    def _from_dict_expanded(cls, item_dict: dict, **kwargs):
        assert '_typedict' in item_dict

        constructor_params = cls.get_constructor_params()
        constructor_dict = {}
        post_construction_dict = {}
        for key, val in item_dict.items():
            if key in ['_typedict']:
                continue
            if key == '_objects':
                assert type(val) is list
                if len(val) == 0:
                    constructor_dict[key] = val
                else:
                    loaded_val: list = []
                    for val0 in val:
                        assert '_typedict' in val0
                        obj_type = BaseUtil.from_type_dict(val0['_typedict'])
                        loaded_val.append(obj_type.from_dict(val0, **kwargs))
                    constructor_dict[key] = loaded_val
            elif type(val) is dict and '_typedict' in val:
                inner_cls = BaseUtil.from_type_dict(val['_typedict'])
                assert hasattr(inner_cls, 'from_dict')
                inner = inner_cls.from_dict(val)
                if key in constructor_params:
                    constructor_dict[key] = inner
                else:
                    post_construction_dict[key] = inner
            else:
                if key in constructor_params:
                    constructor_dict[key] = val
                else:
                    post_construction_dict[key] = val
        obj = cls(**constructor_dict)
        
        if '_typedict' in item_dict and '_args' in item_dict['_typedict'] and cls.__module__ != 'builtins':
            obj.__dict__['__orig_class__'] = typing._GenericAlias(
                cls,
                tuple([BaseUtil.from_type_dict(arg_dict) for arg_dict in item_dict['_typedict']['_args']])
            )

        for key, val in post_construction_dict.items():
            assert hasattr(obj, key), f"{type(obj).__name__} has no attribute {key}"
            setattr(obj, key, val)
        return obj

    @classmethod
    def from_dict(cls, item_dict: dict, compressed: bool=True, **kwargs):
        if compressed:
            return cls._from_dict_compressed(item_dict, **kwargs)
        else:
            return cls._from_dict_expanded(item_dict, **kwargs)

    def append(self, obj: T):
        self._objects.append(obj)
    
    def remove(self, obj: T):
        self._objects.remove(obj)

    def pop(self, idx: int=None) -> T:
        if idx is None:
            idx = len(self._objects) - 1
        return self._objects.pop(idx)

    def index(self, i: T | Callable[[T], bool]=None, **kwargs) -> int:
        if len(self) == 0:
            raise IndexError(f"{type(self).__name__} is empty.")
        elif i is not None:
            if type(i) is type(self[0]):
                return self._objects.index(i)
            elif callable(i):
                for idx, obj in enumerate(self):
                    if i(obj):
                        return idx
                raise ValueError
            else:
                raise TypeError
        elif len(kwargs) > 0:
            for idx, obj in enumerate(self):
                is_match = True
                for key, val in kwargs.items():
                    if getattr(obj, key) != val:
                        is_match = False
                        break
                if is_match:
                    return idx
        else:
            raise ValueError("Must provide parameters.")

    def shuffle(self):
        random.shuffle(self._objects)

    def save(self, path: str, **kwargs):
        ext = os.path.splitext(path)[1]
        if ext == '.json':
            json.dump(self.to_dict(**kwargs), open(path, 'w'), ensure_ascii=False, sort_keys=False)
        elif ext == '.yaml':
            yaml.dump(self.to_dict(**kwargs), open(path, 'w'), allow_unicode=True, sort_keys=False)
        else:
            raise Exception(f"Invalid file extension: {ext}")
    
    @classmethod
    def load(cls, path: str, **kwargs):
        ext = os.path.splitext(path)[1]
        if ext == '.json':
            return cls.from_dict(json.load(open(path, 'r')), **kwargs)
        elif ext == '.yaml':
            return cls.from_dict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader), **kwargs)
        else:
            raise Exception(f"Invalid file extension: {ext}")

H = TypeVar('H', bound=BaseHandler) # H can only be BaseHandler or a subtype of BaseHandler

class BaseObject(Base):
    def __init__(self):
        super().__init__()
        self.id = uuid.uuid4()
    
    def to_dict(self, **kwargs) -> dict:
        item_dict = super().to_dict(**kwargs)
        item_dict['id'] = item_dict['id'].hex
        return item_dict
    
    @classmethod
    def from_dict(cls, item_dict: dict, **kwargs):
        item_dict['id'] = uuid.UUID(item_dict['id'])
        return super().from_dict(item_dict, **kwargs)

OBJ = TypeVar('OBJ', bound=BaseObject)

class BaseObjectHandler(BaseHandler[OBJ]):
    def __init__(self, _objects: list[OBJ]=None):
        super().__init__(_objects)
        self.id = uuid.uuid4()

    def to_dict(self, **kwargs) -> dict:
        item_dict = super().to_dict(**kwargs)
        item_dict['id'] = item_dict['id'].hex
        return item_dict
    
    @classmethod
    def from_dict(cls, item_dict: dict, **kwargs):
        item_dict['id'] = uuid.UUID(item_dict['id'])
        return super().from_dict(item_dict, **kwargs)

OBJ_H = TypeVar('OBJ_H', bound=BaseObjectHandler)

ContextVar = T | H | OBJ | OBJ_H

class Context:
    """
    The goal of this class is to make it possible for a single serialized variable can be referenced in multiple places.
    For example, we could define a LoadableImageMaskHandler and use it to randomly overlay an image on a working image.
    We may then decide to use the same LoadableImageMaskHandler instance to randomly overlay another image on the working image after doing a rotation.
    When we serialize the two calls to the overlay_image function, we would need to serialize the LoadableImageMaskHandler instance as well.
    Without a context, we would end up serializing the same instance twice, resulting in a larger dump file.
    In order to mitigate this problem, instead of serializing the LoadableImageMaskHandler directly in the overlay_image function callback,
    we could serialize it in the context and reference a unique id in the overlay_image function callback instead. 
    """
    def __init__(self, _vars: dict[uuid.UUID, ContextVar]=None):
        self._vars: dict[uuid.UUID, ContextVar] = _vars if _vars is not None else {}
    
    def to_dict(self) -> dict:
        return dict(
            _typedict=BaseUtil.to_type_dict(type(self)),
            _vars={
                key.hex: val.to_dict()
                for key, val in self._vars.items()
            }
        )

    @classmethod
    def from_dict(cls, item_dict: dict) -> Context:
        _vars: dict[uuid.UUID, ContextVar] = {}
        for key, val in item_dict['_vars'].items():
            assert type(val) is dict and '_typedict' in val
            cls_type = BaseUtil.from_type_dict(val['_typedict'])
            assert hasattr(cls_type, 'from_dict')
            cls_obj = cls_type.from_dict(val)
            _vars[uuid.UUID(key)] = cls_obj
        return Context(_vars=_vars)

    def register_variable(self, var: ContextVar) -> ContextVarRef:
        varId = uuid.uuid4()
        self._vars[varId] = var
        return ContextVarRef(varId=varId, context=self)
    
    def unregister_variable(self, value: uuid.UUID | ContextVarRef):
        if type(value) is uuid.UUID:
            varId = value
        elif type(value) is ContextVarRef:
            varId = value.varId
        else:
            raise TypeError
        del self._vars[varId]
    
    def clear(self):
        self._vars.clear()

class ContextVarRef:
    """This is a reference to a variable in the context.
    It is meant to be used together with an existing context.
    """
    def __init__(self, varId: uuid.UUID, context: Context):
        self.varId = varId
        self.context = context

    def to_dict(self) -> dict:
        return dict(
            _typedict=BaseUtil.to_type_dict(type(self)),
            varId=self.varId.hex
        )

    @classmethod
    def from_dict(cls, item_dict: dict, context: Context) -> ContextVarRef:
        return ContextVarRef(
            varId=uuid.UUID(item_dict['varId']),
            context=context
        )

    @property
    def value(self) -> ContextVar:
        return self.context._vars[self.varId]
