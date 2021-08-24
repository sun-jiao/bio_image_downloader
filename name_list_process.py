# for processing files like this: https://www.douban.com/group/topic/30636732/

def get_rank_level(strs0, strs1, str2) -> (int, str, str):
    """
    :param strs0: strs[0]  in split result
    :param strs1:  strs[1] in split result
    :param str2:  strs[2] in split result
    :return: level, zh name, latin name
    """
    if strs0.endswith('总科'):
        return 0, strs0, strs1
    elif strs0 == 'Superfamily':
        return 0, '', strs1
    elif strs0.endswith('亚科'):
        return 2, strs0, strs1
    elif strs0 == 'Subfamily':
        return 2, '', strs1
    elif strs0.endswith('科'):
        return 1, strs0, strs1
    elif strs0 == 'Family':
        return 1, '', strs1
    elif strs0.endswith('族'):
        return 3, strs0, strs1
    elif strs0 == 'Tribe':
        return 3, '', strs1
    elif strs0.endswith('亚属'):
        return 5, strs0, strs1
    elif strs0 == 'Subgenus':
        return 5, '', strs1
    elif strs0.endswith('属'):
        return 4, strs0, strs1
    elif strs0 == 'Genus':
        return 4, '', strs1
    elif strs0.endswith('蝶'):
        return 6, strs0, strs1 + ' ' + str2
    else:
        return 6, '', strs0 + ' ' + strs1


class Taxonomy(object):
    def __init__(self):
        self.rank_levels = [('', '', '')] * 7  # zh, latin, author
        # supfam = 0
        # fam = 1
        # subfam = 2
        # tribe = 3
        # genus = 4
        # subg = 5
        # sp = 6
        self.syn = ''
        self.rank_names = ['Superfamily', 'Family', 'Subfamily', 'Tribe', 'Genus', 'Subgenus', '']
        super(Taxonomy, self).__init__()

    def update(self, level, zh, latin, line):
        # 删除中文名+空格和学名+空格，结果一般就是命名人引证
        author = line.replace(latin + '', '')
        if zh == '' and level != 6:
            author = author.replace(self.rank_names[level] + ' ', '')
        elif zh != '':
            author = author.replace(zh + ' ', '')

        # 如果结果含有 ' (='，则应提取异名。
        if level == 6 and ' (=' in author:
            author, self.syn = author.split(' (=')
            self.syn = self.syn.replace('[', '').replace(']', '').replace(')', '').strip()
        else:
            self.syn = ''

        # 更新相应的层级及所有下级层级。
        self.rank_levels[level] = (zh.strip(), latin.strip(), author.strip())
        idx = level + 1
        while idx < 7:
            self.rank_levels[idx] = ('', '', '')
            idx = idx + 1

    def __str__(self):
        string = ''
        for level in self.rank_levels:
            for i in level:
                string = string + '"' + i + '"' + ','

        string = string + '"' + self.syn + '"'
        return string


def from_csv_name_list(filename, outname):
    taxonomy = Taxonomy()
    out_csv_file = open(outname, 'a', encoding='gbk')
    with open(filename) as file:
        for line in file:
            strs = line.split(' ')
            if len(strs) < 3:
                strs.extend(['']*3)
            level, zh, latin = get_rank_level(strs[0], strs[1], strs[2])
            taxonomy.update(level, zh, latin, line)
            print(taxonomy)
            if level == 6:
                out_csv_file.write(str(taxonomy) + '\r')


if __name__ == '__main__':
    from_csv_name_list('china-butterfly-list-by-Wenhao.txt', 'china-butterfly-list-by-Wenhao.csv')
