import subprocess
import os
import numpy as np

#Esse código utiliza o programa FFmpeg, muito utilizado para manipulação de video e audio,
#para aplicar filtro de crop(corte) em videos a partir de uma determinada pasta.
# Ele utiliza a biblioteca OS do python para mapear os arquivos e percorre cada
#arquivo com a extensão .mov que pertence ao dataset selecionado. Também utiliza a biblioteca subprocess
#que permite a execução de comandos externos a linguagem, como se estivesse rodando no prompt de comando
#do Windos(cmd)
pasta = 'D:\MestradoCEFET\Topicos\Base_Videos_Versao1\Separados\Todos'
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
#Cria uma lista com o caminho dos arquivos
video = [arq for arq in arquivos if arq.lower().endswith(".mov")]

print (len(video))
#Executa cada linha de comando para realizar o crop nos arquivos de videos
for a in video:
    comando = ('ffmpeg.exe -i ' + a + ' -filter:v "crop=273:330:0:0" -c:a copy '+ a +'_Crop_'+'OutputVideo.avi')
    p = subprocess.call(comando, shell=True)


