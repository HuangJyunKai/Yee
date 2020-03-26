import cv2
import os

def save_violation(violation_frame,img_stack,img_result,frame_num):
    if violation_frame != -1 :
        img_stack.append(img_result)
        f = frame_num
        index = [0,f//2,f,int(f*1.5),f*2]
        # 且湊滿frame_num*2張frame，更改旗標，清空
        if len(img_stack) == f*2+1 :
            tmp_path = './violation/'+str(violation_frame)+'/'
            for i in index :
                save_index = violation_frame-f+i
                img_path = tmp_path + str(save_index)+'.png'
                cv2.imwrite(img_path,img_stack[i])
            print('==============>>>>>>  Sucess save Violation Image folder : ',violation_frame)
            f = frame_num*-1
            img_stack = img_stack[f:]
            violation_frame = -1
    # 如果沒有違規        
    else : 
        if len(img_stack) < frame_num :
            img_stack.append(img_result)
        else:
            f = (frame_num*-1)+1
            img_stack = img_stack[f:] 
            img_stack.append(img_result) #只保留最新frame_num筆img_result
    return violation_frame,img_stack
    
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('==============>>>>>>  Scuessfully Make folder :' + directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
