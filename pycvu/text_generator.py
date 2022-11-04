import random
from .base import Base

class TextGenerator(Base):
    def __init__(self, characters: str, textLength: int, allowRepetition: bool=True):
        self.characters = characters
        self.textLength = textLength
        self.allowRepetition = allowRepetition
    
    def random(self) -> str:
        indices: list[int] = []
        characters = list(self.characters)
        if self.allowRepetition:
            indices: list[int] = [random.randint(0, len(characters)-1) for i in range(self.textLength)]
        else:
            assert len(self.characters) >= self.textLength
            indices: list[int] = []
            availableIndices: list[int] = list(range(len(characters)))
            for i in range(self.textLength):
                j = random.randint(0, len(availableIndices)-1)
                indices.append(availableIndices[j])
                del availableIndices[j]
        return ''.join([characters[idx] for idx in indices])

    def debug():
        print(TextGenerator('abcdefghijklmnopqrstuvwxyz', 26, allowRepetition=False).random())
