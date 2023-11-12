解压含中文的压缩包：`unzip -O utf-8 data.zip`

AutoDL服务器不支持中文的解决方法：
`locale` 查看服务器编码
`export LANG=zh_CN.UTF-8`将编码改为中文
