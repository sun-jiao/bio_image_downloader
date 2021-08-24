文件说明：

###china-butterfly-list-by-Wenhao.txt:

[豆瓣用户sunwenhao整理的中国蝴蝶名录](https://www.douban.com/group/topic/30636732/) 的纯文字版本,该文件著作权属于原作者，本仓库的开源协议不适用。

###china-butterfly-list-by-Wenhao.csv:

上述文件经过name_list_process.py处理后的名录，包含每个物种及其上级分类级别的中文名、学名、命名人和异名

###china-butterfly-list-by-Wenhao.xlsx:

上述文件的excel版本

###download_list.csv:

用于下载图片。

基本结构：folder, latin1, zh1, latin2, zh2, ...

###label.csv:

label 和 class name 之间的对应文件

###taxonloss_config.csv

用于计算分类学损失函数的配置文件。

基本结构：label, rank1, rank2, rank3, rank4, (from higher to lower)..., species name
