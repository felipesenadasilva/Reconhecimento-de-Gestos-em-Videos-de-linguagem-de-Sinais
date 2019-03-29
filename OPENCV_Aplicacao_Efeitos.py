import os
import cv2
import numpy as np

pasta = 'D:\MestradoCEFET\Topicos\Base_Videos_Versao1\Separados\Above_acima\cropados'
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
video = [arq for arq in arquivos if arq.lower().endswith(".avi")]

fps = 15

##APLICACAO DE Blur ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_Blur_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
    
        blur = cv2.GaussianBlur(frame,(15,15),0)    
        cv2.imshow('Original',frame)
        cv2.imshow('Gaussian Blurring',blur)
        videoWriter.write(blur)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()

##APLICACAO DE Media Blur ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_MedianBlur_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
    
        median = cv2.medianBlur(frame,15)
        cv2.imshow('Original',frame)
        cv2.imshow('Median Blur',median)
        videoWriter.write(median)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()
    
##APLICACAO DE Smoothed ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_Smoothed_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break

        kernel = np.ones((15,15),np.float32)/225
        smoothed = cv2.filter2D(frame,-1,kernel)    
        cv2.imshow('Original',frame)
        cv2.imshow('Smoothed',smoothed)

        videoWriter.write(smoothed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()
    
##APLICACAO DE linha ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_Linha_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break

        linha = cv2.line(frame,(0,0),(200,300),(255,255,255),50)
        cv2.imshow('Original',frame)
        cv2.imshow('Linha',linha)

        videoWriter.write(linha)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()
    
##APLICACAO DE Retangulo ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_Retangulo_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break

        retangulo = cv2.rectangle(frame,(50,25),(100,50),(0,0,255),15)
        cv2.imshow('Original',frame)
        cv2.imshow('retangulo',retangulo)

        videoWriter.write(retangulo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()
    
    
##APLICACAO DE Circulo ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_Circulo_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break

        circulo = cv2.circle(frame,(100,30), 5, (0,255,0), -1)
        cv2.imshow('Original',frame)
        cv2.imshow('circulo',circulo)

        videoWriter.write(circulo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()

##APLICACAO DE polilinhas ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_polilinhas_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        
        pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
        pts = pts.reshape((-1,1,2))
        polilinhas = cv2.polylines(frame, [pts], True, (0,255,255), 3)
        
        cv2.imshow('Original',frame)
        cv2.imshow('polilinhas',polilinhas)

        videoWriter.write(polilinhas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()

##APLICACAO DE Texto ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_Texto_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        texto = cv2.putText(frame,'OK',(10,100), font, 4, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.imshow('Original',frame)
        cv2.imshow('Texto',texto)

        videoWriter.write(texto)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()

##APLICACAO DE Rotacao ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_Rotacao_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        
        rotacao = cv2.flip (frame, 0)
        
        cv2.imshow('Original',frame)
        cv2.imshow('Rotacao',rotacao)

        videoWriter.write(rotacao)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()
    

##APLICACAO DE HSV ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_HSV_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        cv2.imshow('Original',frame)
        cv2.imshow('hsv',hsv)

        videoWriter.write(hsv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()  

    
##APLICACAO DE brilho ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_brilho_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        
        brilho = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        brilho[:,:,2] = brilho[:,:,2]*random_bright
        brilho = cv2.cvtColor(brilho,cv2.COLOR_HSV2RGB)
        
        cv2.imshow('Original',frame)
        cv2.imshow('brilho',brilho)

        videoWriter.write(brilho)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows() 
    
##APLICACAO DE Erosao ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_erosao_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        erosion = cv2.erode(hsv,None,iterations = 2)

        
        cv2.imshow('Original',frame)
        cv2.imshow('erosao',erosion)

        videoWriter.write(erosion)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows() 
    

##APLICACAO DE Dilatacao ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_Dilatacao_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dilation = cv2.dilate(hsv,None,iterations = 2)

        
        cv2.imshow('Original',frame)
        cv2.imshow('Dilatacao',dilation)

        videoWriter.write(dilation)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows() 

##APLICACAO DE Translacao ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_Translacao_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        
        trans_range=5
        
        #Translacao
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        
        (h, w) = frame.shape[:2]

        translacao = cv2.warpAffine(frame,Trans_M, (w, h))
        
        cv2.imshow('Original',frame)
        cv2.imshow('Translacao',translacao)

        videoWriter.write(translacao)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows() 
    

##APLICACAO DE Rotacao ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_rotacao_2_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        
        ang_range=20
        
        #Rotacao
        (h, w) = frame.shape[:2]
        center = (w / 2, h / 2)
        ang_rot = np.random.uniform(ang_range)-ang_range/2    
        Rot_M = cv2.getRotationMatrix2D(center, ang_rot, 1)
        
        rotacao = cv2.warpAffine(frame, Rot_M, (w, h))  
        
        cv2.imshow('Original',frame)
        cv2.imshow('Rotacao 2',rotacao)

        videoWriter.write(rotacao)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()
    

##APLICACAO DE Corte ##
for a in video:
   
    cap = cv2.VideoCapture(a)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(a + '_Corte_'+'OutputVideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        
        shear_range=10
        
        #Corte
        pt1 = 5+shear_range*np.random.uniform()-shear_range/2
        pt2 = 20+shear_range*np.random.uniform()-shear_range/2
        pts1 = np.float32([[5,5],[20,5],[5,20]])
        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
        shear_M = cv2.getAffineTransform(pts1,pts2) 

        corte = cv2.warpAffine(frame,shear_M, (w, h))
        
        cv2.imshow('Original',frame)
        cv2.imshow('corte',corte)

        videoWriter.write(corte)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()
