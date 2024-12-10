from typing import Any, Union


class OrderedSet:
    def __init__(self) -> None:
        self.order = ['unknow']
        self.unique_elements = set()
    
    def add(self, x: Union[int, float, str]) -> None:
        if x not in self.unique_elements:
            self.order.remove('unknow')
            self.order.append(x)
            self.order.append('unknow')
            self.unique_elements.add(x)
    
    def remove(self, x: Any) -> None:
        if x in self.order:
            self.order.remove(x)
            self.unique_elements.remove(x)
    
    def clear(self) -> None:
        self.order.clear()
        self.unique_elements.clear()
        
    def index(self, x: Any) -> None:
        if x in self.order:
            return self.order.index(x)
        else:
            raise ValueError("f{x} not found in OrderedSet")
    
    def __len__(self):
        return len(self.order)

    def __str__(self):
        return str(self.order)
    
    def __getitem__(self, index: int):
        return self.order[index]