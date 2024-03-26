'''from selenium import webdriver
from selenium.webdriver.common.by import By
#打开谷歌浏览器
driver = webdriver.Chrome()
#打开百度搜索主页
driver.get('https://www.baidu.com')
driver.find_element(By.XPATH,'//*[@id="kw"]').send_keys('爱奇艺')
driver.find_element(By.XPATH,'//*[@id="su"]').click()'''
import os
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import shutil
from selenium.webdriver.chrome.options import Options

def download_video(url):
    #chrome_options = Options()
    #chrome_options.add_argument('--headless')
    driver = webdriver.Chrome()
    driver.get('https://savetwitter.net/zh-cn#google_vignette')
    # send_keys设置input框内容  click处理点击
    time.sleep(3)
    driver.find_element(By.XPATH, '//*[@id="s_input"]').send_keys(url)
    time.sleep(3)
    driver.find_element(By.XPATH, '//*[@id="search-form"]/button').click()
    time.sleep(3)
    down_url = driver.find_element(By.XPATH, '//*[@id="search-result"]/div[2]/div[2]/div/p[1]/a').get_attribute('href')
    print(down_url)
    driver.get(down_url)
    time.sleep(12)
    driver.quit()


def download_photo(url):
    #chrome_options = Options()
    #chrome_options.add_argument('--headless')
    driver = webdriver.Chrome()
    driver.get('https://savetwitter.net/zh-cn#google_vignette')
    # send_keys设置input框内容  click处理点击
    time.sleep(3)
    driver.find_element(By.XPATH, '//*[@id="s_input"]').send_keys(url)
    time.sleep(3)
    driver.find_element(By.XPATH, '//*[@id="search-form"]/button').click()
    time.sleep(3)
    down_url = driver.find_element(By.XPATH, '//*[@id="search-result"]/div[2]/div/ul/li/div/div[2]/a').get_attribute('href')
    print(down_url)
    driver.get(down_url)
    time.sleep(8)
    driver.quit()


def move(people):
    path = 'C:\\Users\\lenovo\\Downloads'  # 要遍历的目录
    for root, dirs, names in os.walk(path):
        for name in names:
            ext = os.path.splitext(name)[1]  # 获取后缀名
            if ext == '.mp4':
                fromdir = os.path.join(root, name)  # mp4文件原始地址
                moveto = os.path.join('./file/'+people+'_video', name)  ##dirname 上一层目录
                shutil.move(fromdir, moveto)  # 移动文件
            elif ext == '.jpg' or ext == '.png':
                fromdir = os.path.join(root, name)  # mp4文件原始地址
                moveto = os.path.join('./file/'+people+'_photo', name)  ##dirname 上一层目录
                shutil.move(fromdir, moveto)  # 移动文件


def num_video():
    num = 0
    path = 'C:\\Users\\lenovo\\Downloads'  # 要遍历的目录
    for root, dirs, names in os.walk(path):
        for name in names:
            ext = os.path.splitext(name)[1]  # 获取后缀名
            if ext == '.mp4':
                num = num+1
    return num


def num_photo():
    num = 0
    path = 'C:\\Users\\lenovo\\Downloads'  # 要遍历的目录
    for root, dirs, names in os.walk(path):
        for name in names:
            ext = os.path.splitext(name)[1]  # 获取后缀名
            if ext == '.jpg' or ext == '.png':
                num = num+1
    return num


f = open("redirect_url.txt", "r", encoding='utf-8')
people = 246
for line in f.readlines()[16236:]:
    line = line[0:-1]
    print(line)
    if line == str(people):
        os.mkdir('./file/'+str(people)+'_video')
        os.mkdir('./file/'+str(people)+'_photo')
        move(str(people))
        print('第 %d 个用户的网址读取完毕' %people)
        people = people + 1
        print('开始读取第 %d 个用户的网址' %people)
    elif line == 'none':
        print('第 %d 个用户为none' % people)
    else:
        num_vi = num_video()
        num_ph = num_photo()
        print('num_video: %d ' %num_vi)
        print('num_photo: %d ' %num_ph)

        if line[-7:-2] == 'video':
            if(num_vi < 20):
                try:
                    download_video(line)
                    print('视频下载成功')
                except:
                    print('视频下载失败')
                    continue

        elif line[-7:-2] == 'photo':
            if(num_ph < 20):
                try:
                    download_photo(line)
                    print('图片下载成功')
                except:
                    print('图片下载失败')
                    continue
        else:
            if (num_video() < 20):
                try:
                    download_video(line)
                    print('视频下载成功')
                    continue
                except:
                    print('该网址失效或无视频')
            if (num_photo() < 20):
                try:
                    download_photo(line)
                    print('图片下载成功')
                except:
                    print('该网址失效或无图片')
            continue
f.close()
