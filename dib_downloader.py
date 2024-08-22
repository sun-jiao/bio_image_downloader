for a in range(1, 71):
    # print(f'nohup wget http://dongniao.cn-bj.ufileos.com/DIB-10K%2F{a}.tgz &')
    print(f'tar -xvzf DIB-10K%2F{a}.tgz -C dib && rm DIB-10K%2F{a}.tgz')
