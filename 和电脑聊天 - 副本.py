import time
import requests
import urllib
import json

from pyaudio import PyAudio,paInt16
import wave #自带的模块
import base64
import pyttsx3

def record_audio():
	pa = PyAudio()
	audio_eq = pa.open(
		format = paInt16, #音频的存储位深
		channels = 1, #声道值
		rate = 16000, #采样率
		input = True, #输入
		frames_per_buffer = 1024, #获取的数据大小
	)#打开音频设备

	times = 0
	data = []
	print('请说话')
	start = time.time()
	while times<50:
		audio_data = audio_eq.read(1024)
		data.append(audio_data)
		times += 1
	end = time.time()
	print('说完了:%.2f' % (end-start))	
	audio_eq.close()
	data = b''.join(data)
	return data

def baidu_anylist():
	#获取音频
	audio_data = record_audio()
	audio_len = len(audio_data)
	speech = base64.b64encode(audio_data).decode()

	bd_url = 'http://vop.baidu.com/server_api'
	data = {
		'format':'wav',
		'rate': 16000,
		'channel':1,
		'cuid':'Ssdasd',
		'token':'',
		'speech':speech,
		'len':audio_len,
	}

	data = json.dumps(data).encode()
	headers = {
	'Content-type':'application/json',
	}

	req = requests.post(url=bd_url, data=data, headers=headers)
	req_json = req.json() #req为response格式 使用json()转为字典格式
	print(req_json) #req.text 为str格式

	if(req_json['err_msg'] == 'success.'):
		print('[question] ' + req_json['result'][0])
		return req_json['result'][0]
	else:
		print('[info] 报错了')
	# 成功返回 {"corpus_no":"6548720418950732269","err_msg":"success.","err_no":0,"result":["你好，"],"sn":"35484909961524742790"}

	#req = urllib.request.Request(url=url, data=data, headers=headers)
	#result = json.loads(urllib.request.urlopen(req).read().decode('gb2312','ignore'))
	
def say_hello():
	eg = pyttsx3.init()
	print('[+] 计算机说:')
	eg.say('Hello,it\'s me')
	eg.runAndWait()
	print('[+] 计算机说完了……')

def play_say(data):
	eg = pyttsx3.init()
	eg.say(data)
	eg.runAndWait()

def tl_callback(question):
	tl_url = 'http://openapi.tuling123.com/openapi/api'
	api_key = ''
	loc = '南京市'
	r= requests.get(url=tl_url , params={"key":api_key,"loc":loc,"info":question})
	answer = json.loads(r.text)['text']
	print('[answer] ' +answer)
	return answer

if __name__ == '__main__':
	while True:
		#say_hello()
		char = input('\r\n是否输入问题 ')
		if char == 'y':
			# play_say('当然是选择原谅他')
			question = baidu_anylist()
			answer = tl_callback(question)
			play_say(answer)
		else:
			print('再见')
			play_say('再见')
			break




