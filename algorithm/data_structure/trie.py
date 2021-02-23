"""
字典树：
    又称单词查找树，是一种树形结构，是一种哈希树的变种。
    典型应用于统计，排序和保存大量的字符串，所以经常被搜索引擎系统用于文本词频统计。
    优点：利用字符串的公共前缀来减少查询时间，最大限度地减少无畏的字符串比较，查询效率比哈希树高。
    性质：
        1. 根节点不包含字符，除根节点外每个节点都只包含一个字符；
        2. 从根节点到某一节点，路径上经过的字符连接起来，为该节点对应的字符串；
        3. 每个节点的所有子节点包含的字符都不相同。
"""


class TrieNode(object):
    def __init__(self):
        self.nodes = dict()
        self.is_leaf = False

    def insert(self, word: str):
        """
        插入一个字到字典树中。
        :param word:
        :return:
        """
        curr = self
        for char in word:
            if char not in curr.nodes:
                curr.nodes[char] = TrieNode()
            curr = curr.nodes[char]
        curr.is_leaf = True

    def insert_many(self, words: [str]):
        """
        插入一列表的字到字典中。
        :param words:
        :return:
        """
        for word in words:
            self.insert(word)

    def search(self, word: str):
        """
        在字典树里查询一个字。
        :param word:
        :return:
        """
        curr = self
        for char in word:
            if char not in curr.nodes:
                return False
            curr = curr.nodes[char]
        return curr.is_leaf


