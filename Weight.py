import numpy as np
import torch
from Config import class_num

def convert_to_onehot(sca_label, class_num=31):
    return np.eye(class_num)[sca_label]

# lmmd的权重矩阵，代码本身就有的，注释是自己加的
class Weight:

    @staticmethod
    def cal_weight(s_label, t_label, type='visual', batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()   # 1*batchsize的标签向量 eg:[0,5,3,2,4,1,2,6]
        s_vec_label = convert_to_onehot(s_sca_label, class_num=class_num)  # 转换为独热编码，batchsize*numcls
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)  # 对标签求和
        s_sum[s_sum == 0] = 100  # 猜测是避免除以0，将分母0改成100，应该可以随意设置，因为0/任何数都是0
        s_vec_label = s_vec_label / s_sum  # 生成权重Ws

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()  # 1*batchsize的标签向量 eg:[0,5,3,2,4,1,2,6]
        #t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()  # batchsize*numcls，每一行里都是标签的概率分布
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)  # 对标签求和
        t_sum[t_sum == 0] = 100  # 应该只是为了和目标域一致，毕竟概率分布应该很小几率出现0
        t_vec_label = t_vec_label / t_sum  # 生成权重Wt

        weight_ss = np.zeros((batch_size, batch_size))  # 生成权重矩阵，i*j
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)  # 返回一个集合，重复元素只保留一个
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:  # 这里好像感觉是只考虑了一个batch中两个域中都有的类别
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)  # 抽取存在的类别对应的B个样本的权重向量Wi
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)  # 抽取存在的类别对应的B个样本的权重向量Wt
                ss = np.dot(s_tvec, s_tvec.T)  # 生成Ws*Ws的权重矩阵
                weight_ss = weight_ss + ss  # / np.sum(s_tvec) / np.sum(s_tvec)  # 权重矩阵相加
                tt = np.dot(t_tvec, t_tvec.T)  # 生成Wt*Wt的权重矩阵
                weight_tt = weight_tt + tt  # / np.sum(t_tvec) / np.sum(t_tvec)  # 权重矩阵相加
                st = np.dot(s_tvec, t_tvec.T)  # 生成Ws*Wt的权重矩阵
                weight_st = weight_st + st  # / np.sum(s_tvec) / np.sum(t_tvec)  # 权重矩阵相加
                count += 1  # C

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length  # /C
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


# nmmd的权重矩阵，根据Weight修改的
# 主要是修改了s_vec_label、t_vec_label，将各种情感分类变成正负分类
class Weight_1:

    @staticmethod
    def cal_weight(s_label, t_label, mode, type='visual', batch_size=32, class_num=2):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()   # 1*batchsize的标签向量 eg:[0,2,3,2,4,1,2,2]
        # 0负面，1正面
        # 先把负面的情感都标5，正面都标6，最后把标5的都标成0，标6的都标成1
        # 这样就实现了0负面，1正面
        # [0,2,3,2,4,1,2,2] -> [0,0,1,0,0,0,0,0]（be）
        if mode == 'be':  # [angry, disgust, fear, happy, sad]
            s_sca_label[s_sca_label == 0] = 5
            s_sca_label[s_sca_label == 1] = 5
            s_sca_label[s_sca_label == 2] = 5
            s_sca_label[s_sca_label == 3] = 6
            s_sca_label[s_sca_label == 4] = 5
            s_sca_label[s_sca_label == 5] = 0
            s_sca_label[s_sca_label == 6] = 1
        elif mode == 'bc':  # [angry, fear, happy, neutral, sad]
            s_sca_label[s_sca_label == 0] = 5
            s_sca_label[s_sca_label == 1] = 5
            s_sca_label[s_sca_label == 2] = 6
            s_sca_label[s_sca_label == 3] = 6
            s_sca_label[s_sca_label == 4] = 5
            s_sca_label[s_sca_label == 5] = 0
            s_sca_label[s_sca_label == 6] = 1
        elif mode == 'ec':  # [angry, fear, happy, sad, surprise]
            s_sca_label[s_sca_label == 0] = 5
            s_sca_label[s_sca_label == 1] = 5
            s_sca_label[s_sca_label == 2] = 6
            s_sca_label[s_sca_label == 3] = 5
            s_sca_label[s_sca_label == 4] = 6
            s_sca_label[s_sca_label == 5] = 0
            s_sca_label[s_sca_label == 6] = 1
        s_vec_label = convert_to_onehot(s_sca_label, class_num=class_num)  # 转换为独热编码，batchsize*numcls
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)  # 对标签求和
        s_sum[s_sum == 0] = 100  # 猜测是避免除以0，将分母0改成100，应该可以随意设置，因为0/任何数都是0
        s_vec_label = s_vec_label / s_sum  # 生成权重Ws

        # t_vec_label里每一行代表一个样本，每一个样本分正负两类，这里先初始化
        # t_vec_label = np.zeros((8, 2))
        t_vec_label = np.zeros((32, 2))
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()  # 1*batchsize的标签向量 eg:[0,5,3,2,4,1,2,6]
        #t_vec_label = convert_to_onehot(t_sca_label)
        if mode == 'be':  # [angry, disgust, fear, happy, sad]
            t_sca_label[t_sca_label == 0] = 5
            t_sca_label[t_sca_label == 1] = 5
            t_sca_label[t_sca_label == 2] = 5
            t_sca_label[t_sca_label == 3] = 6
            t_sca_label[t_sca_label == 4] = 5
            t_sca_label[t_sca_label == 5] = 0
            t_sca_label[t_sca_label == 6] = 1
        elif mode == 'bc':  # [angry, fear, happy, neutral, sad]
            t_sca_label[t_sca_label == 0] = 5
            t_sca_label[t_sca_label == 1] = 5
            t_sca_label[t_sca_label == 2] = 6
            t_sca_label[t_sca_label == 3] = 6
            t_sca_label[t_sca_label == 4] = 5
            t_sca_label[t_sca_label == 5] = 0
            t_sca_label[t_sca_label == 6] = 1
        elif mode == 'ec':  # [angry, fear, happy, sad, surprise]
            t_sca_label[t_sca_label == 0] = 5
            t_sca_label[t_sca_label == 1] = 5
            t_sca_label[t_sca_label == 2] = 6
            t_sca_label[t_sca_label == 3] = 5
            t_sca_label[t_sca_label == 4] = 6
            t_sca_label[t_sca_label == 5] = 0
            t_sca_label[t_sca_label == 6] = 1
        t_vec_label_temp = t_label.cpu().data.numpy()  # batchsize*numcls，每一行里都是标签的概率分布
        # t_vec_label_temp里是目标域样本的按照情感分类的概率分布，下面要转换成正负两类的概率分布
        # (8,5) -> (8,2)
        # mode=be时，将t_vec_label_temp中第0、1、2、4列的概率加起来，赋值给t_vec_label第0列,作为负情感的概率
        # 将t_vec_label_temp中第3列的概率作为正情感的概率，赋值给t_vec_label第1列
        # 其他模式依次类推
        if mode == 'be':  # [angry, disgust, fear, happy, sad]
            t_vec_label[:, 0] = t_vec_label_temp[:, 0] + t_vec_label_temp[:, 1] + t_vec_label_temp[:, 2] + t_vec_label_temp[:, 4]
            t_vec_label[:, 1] = t_vec_label_temp[:, 3]
        elif mode == 'bc':  # [angry, fear, happy, neutral, sad]
            t_vec_label[:, 0] = t_vec_label_temp[:, 0] + t_vec_label_temp[:, 1] + t_vec_label_temp[:, 3] + t_vec_label_temp[:, 4]
            t_vec_label[:, 1] = t_vec_label_temp[:, 2]
        elif mode == 'ec':  # [angry, fear, happy, sad, surprise]
            t_vec_label[:, 0] = t_vec_label_temp[:, 0] + t_vec_label_temp[:, 1] + t_vec_label_temp[:, 3]
            t_vec_label[:, 1] = t_vec_label_temp[:, 2] + t_vec_label_temp[:, 4]
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)  # 对标签求和
        t_sum[t_sum == 0] = 100  # 应该只是为了和目标域一致，毕竟概率分布应该很小几率出现0
        t_vec_label = t_vec_label / t_sum  # 生成权重Wt

        weight_ss = np.zeros((batch_size, batch_size))  # 生成权重矩阵，i*j
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)  # 返回一个集合，重复元素只保留一个
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)  # 抽取存在的类别对应的B个样本的权重向量Wi
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)  # 抽取存在的类别对应的B个样本的权重向量Wt
                ss = np.dot(s_tvec, s_tvec.T)  # 生成Ws*Ws的权重矩阵
                weight_ss = weight_ss + ss  # / np.sum(s_tvec) / np.sum(s_tvec)  # 权重矩阵相加
                tt = np.dot(t_tvec, t_tvec.T)  # 生成Wt*Wt的权重矩阵
                weight_tt = weight_tt + tt  # / np.sum(t_tvec) / np.sum(t_tvec)  # 权重矩阵相加
                st = np.dot(s_tvec, t_tvec.T)  # 生成Ws*Wt的权重矩阵
                weight_st = weight_st + st  # / np.sum(s_tvec) / np.sum(t_tvec)  # 权重矩阵相加
                count += 1  # C

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length  # /C
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')

