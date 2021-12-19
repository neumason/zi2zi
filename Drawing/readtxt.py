#!/usr/bin/env Python
#coding=utf-8

import chardet
import sys
import codecs
import json
f2=codecs.open("111.txt","r",'utf-8')
list=[]
for i in range(1077):
    lines=f2.read(1)
    list.append(lines)
print(len(list))
dict={'gbk':list}
jStr=json.dumps(dict)
with open('text1077.json','wb')as f:f.write(jStr)
