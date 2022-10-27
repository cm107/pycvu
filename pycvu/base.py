from __future__ import annotations
import sys
import copy
import inspect
import json
import os
from typing import Any, Callable, Generator, Generic, TypeVar
import uuid
import importlib

import yaml

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

class Base:
    """Base Class
    Assume that all class variables are in the parameter list of __init__
    """
    def __init__(self):
        self._module = type(self).__module__
        self._qualname = type(self).__qualname__

    def __str__(self) -> str:
        result = type(self).__name__
        result += "("
        param_str_list = [f"{key}={val}" for key, val in self.__dict__.items() if key not in ['_module', '_qualname']]
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

    def to_dict(self) -> dict:
        return {
            key: (
                val if not hasattr(val, 'to_dict')
                else val.to_dict()
            )
            for key, val in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict):
        assert '_module' in item_dict
        assert '_qualname' in item_dict
        loaded_cls = getattr(importlib.import_module(item_dict['_module']), item_dict['_qualname'])
        assert cls is loaded_cls, f"{cls=} is not {loaded_cls=}"
        constructor_params = cls.get_constructor_params()
        constructor_dict = {}
        post_construction_dict = {}
        for key, val in item_dict.items():
            if key in ['_module', '_qualname']:
                continue
            if type(val) is dict and '_module' in val and '_qualname' in val:
                inner_cls = getattr(importlib.import_module(val['_module']), val['_qualname'])
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
        for key, val in post_construction_dict.items():
            assert hasattr(obj, key)
            setattr(obj, key, val)
        return obj

    def save(self, path: str):
        ext = os.path.splitext(path)[1]
        if ext == '.json':
            json.dump(self.to_dict(), open(path, 'w'), ensure_ascii=False, sort_keys=False)
        elif ext == '.yaml':
            yaml.dump(self.to_dict(), open(path, 'w'), allow_unicode=True, sort_keys=False)
        else:
            raise Exception(f"Invalid file extension: {ext}")
    
    @classmethod
    def load(cls, path: str):
        ext = os.path.splitext(path)[1]
        if ext == '.json':
            return cls.from_dict(json.load(open(path, 'r')))
        elif ext == '.yaml':
            return cls.from_dict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))
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
        self._module = type(self).__module__
        self._qualname = type(self).__qualname__
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

    def to_dict(self) -> dict:
        item_dict = {}
        for key, val in self.__dict__.items():
            if key == '_objects':
                item_dict[key] = [val0.to_dict() for val0 in val]
            elif hasattr(val, 'to_dict'):
                item_dict[key] = val.to_dict()
            else:
                item_dict[key] = val
        return item_dict

    @classmethod
    def from_dict(cls, item_dict: dict):
        assert '_module' in item_dict
        assert '_qualname' in item_dict
        loaded_cls = getattr(importlib.import_module(item_dict['_module']), item_dict['_qualname'])
        assert cls is loaded_cls, f"{cls=}, {loaded_cls=}"
        constructor_params = cls.get_constructor_params()
        constructor_dict = {}
        post_construction_dict = {}
        for key, val in item_dict.items():
            if key in ['_module', '_qualname']:
                continue
            if key == '_objects':
                assert type(val) is list
                if len(val) == 0:
                    constructor_dict[key] = val
                else:
                    sample = val[0]
                    assert type(sample) is dict
                    assert '_module' in sample
                    assert '_qualname' in sample
                    sample_cls = getattr(importlib.import_module(sample['_module']), sample['_qualname'])
                    assert hasattr(sample_cls, 'from_dict')
                    loaded_val = [sample_cls.from_dict(val0) for val0 in val]
                    constructor_dict[key] = loaded_val
            elif type(val) is dict and '_module' in val and '_qualname' in val:
                inner_cls = getattr(importlib.import_module(val['_module']), val['_qualname'])
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
        for key, val in post_construction_dict.items():
            assert hasattr(obj, key)
            setattr(obj, key, val)
        return obj

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

    def save(self, path: str):
        ext = os.path.splitext(path)[1]
        if ext == '.json':
            json.dump(self.to_dict(), open(path, 'w'), ensure_ascii=False, sort_keys=False)
        elif ext == '.yaml':
            yaml.dump(self.to_dict(), open(path, 'w'), allow_unicode=True, sort_keys=False)
        else:
            raise Exception(f"Invalid file extension: {ext}")
    
    @classmethod
    def load(cls, path: str):
        ext = os.path.splitext(path)[1]
        if ext == '.json':
            return cls.from_dict(json.load(open(path, 'r')))
        elif ext == '.yaml':
            return cls.from_dict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))
        else:
            raise Exception(f"Invalid file extension: {ext}")

class BaseObject(Base):
    def __init__(self):
        super().__init__()
        self.id = uuid.uuid4()
    
    def to_dict(self) -> dict:
        item_dict = super().to_dict()
        item_dict['id'] = item_dict['id'].hex
        return item_dict
    
    @classmethod
    def from_dict(cls, item_dict: dict):
        item_dict['id'] = uuid.UUID(item_dict['id'])
        return super().from_dict(item_dict)

OBJ = TypeVar('OBJ', bound=BaseObject)

class BaseObjectHandler(BaseHandler[OBJ]):
    def __init__(self, _objects: list[OBJ]=None):
        super().__init__(_objects)
        self.id = uuid.uuid4()

    def to_dict(self) -> dict:
        item_dict = super().to_dict()
        item_dict['id'] = item_dict['id'].hex
        return item_dict
    
    @classmethod
    def from_dict(cls, item_dict: dict):
        item_dict['id'] = uuid.UUID(item_dict['id'])
        return super().from_dict(item_dict)

class InnerDummyObj(BaseObject):
    def __init__(self, name: str='Name', count: int=0):
        super().__init__()
        self.name = name
        self.count = count

class DummyObj(BaseObject):
    def __init__(self, a: int, b: int, inner: InnerDummyObj=None):
        super().__init__()
        self.a = a; self.b = b
        self.inner = inner if inner is not None else InnerDummyObj()

    @staticmethod
    def debug():
        
        obj = DummyObj(1, 2)
        print(f"{obj.to_dict()=}")
        print(f"{obj=}")
        obj0 = DummyObj.from_dict(obj.to_dict())
        assert type(obj0.inner) is InnerDummyObj
        print(f"{obj0=}")

class DummyOuter(BaseObject):
    def __init__(self, msg: str='Hello'):
        super().__init__()
        self.msg = msg

class DummyObjHandler(BaseObjectHandler[DummyObj]):
    def __init__(self, _objects: list[DummyObj]=None, dummyOuter: DummyOuter=None):
        super().__init__(_objects)
        self.dummyOuter = dummyOuter if dummyOuter is not None else DummyOuter()
    
    @staticmethod
    def debug():
        handler = DummyObjHandler(
            _objects=[
                DummyObj(1, 2, InnerDummyObj('Fred', 5)),
                DummyObj(1, 2, None)
            ],
            dummyOuter=DummyOuter('How are you?')
        )
        handler0 = DummyObjHandler.from_dict(handler.to_dict())
        print(f"{handler.to_dict()=}")
        print(f"{handler0.to_dict()=}")
        handler0.save('debug_test.json')
        handler1 = DummyObjHandler.load('debug_test.json')
        print(f"{handler1.to_dict()=}")