import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import warnings;warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,Flatten
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

df_train = pd.read_csv('E:/NLP/Kesci2019/Pre_competition/data/train.csv', lineterminator='\n')
df_test = pd.read_csv('E:/NLP/Kesci2019/Pre_competition/data/20190425_test.txt', lineterminator='\n')

df_train['label'] = df_train['label'].map({'Negative':0,'Positive':1})

# 将训练集、测试集转为矩阵形式
numpy_array_train = df_train.as_matrix()
numpy_array_test = df_test.as_matrix()

# nlp语句清洗
def cleaner(word):
    word = re.sub(r'\#\.', '', word)
    word = re.sub(r'\n', '', word)
    word = re.sub(r',', '', word)
    word = re.sub(r'\-', ' ', word)
    word = re.sub(r'\.', '', word)
    word = re.sub(r'\\', ' ', word)
    word = re.sub(r'\\x\.+', '', word)
    word = re.sub(r'\d', '', word)
    word = re.sub(r'^_.', '', word)
    word = re.sub(r'_', ' ', word)
    word = re.sub(r'^ ', '', word)
    word = re.sub(r' $', '', word)
    word = re.sub(r'\?', '', word)
    return word.lower()

def array_cleaner(array):
    # X = array
    X = []
    for sentence in array:
        clean_sentence = ''
        words = sentence.split(' ')
        for word in words:
            clean_sentence = clean_sentence +' '+ cleaner(word)
        X.append(clean_sentence)
    return X

X_train = numpy_array_train[: ,1]
X_test = numpy_array_test[:, 1]

# Clear X here
X_train = array_cleaner(X_train)
X_test = array_cleaner(X_test)

Y_train = numpy_array_train[:, 2]
Y_train = np.array(Y_train)
Y_train = Y_train.astype('int8')

X_all = X_train + X_test
lentrain = len(X_train)

# Tokenizer编码
tokenizer = Tokenizer(nb_words=2000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,split=' ')
tokenizer.fit_on_texts(X_all)

X = tokenizer.texts_to_sequences(X_all)

# pad填充
X = pad_sequences(X)

# embedding + lstm
embed_dim = 128
lstm_out = 256
batch_size = 32

model = Sequential()
model.add(Embedding(2000,embed_dim, input_length=X.shape[1],dropout = 0.5))
model.add(LSTM(256, dropout_U = 0.2, dropout_W = 0.2,return_sequences=True))
model.add(Flatten())
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics = ['accuracy'])
print(model.summary())

# 分回训练集和测试集
X_train = X[:lentrain]
X_test = X[lentrain:]

Y_binary = to_categorical(Y_train)


# RocAuc
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val,self.y_val = validation_data
    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch+1, score))
#             print(y_pred[:6])

x_train5,y_train5,x_label5,y_label5 = train_test_split(X_train,Y_binary, train_size=0.8, random_state=2019)
RocAuc = RocAucEvaluation(validation_data=(y_train5,y_label5), interval=1)

# 训练模型
history = model.fit(x_train5, x_label5, batch_size=batch_size, epochs=5, 
                    validation_data=(y_train5, y_label5), callbacks=[RocAuc], verbose=2)
# # 预测
# Y_lstm = model.predict_proba(X_test,batch_size=batch_size)[:,1]

# lstm_output = pd.DataFrame(data={"ID":df_test["ID"], "Pred":Y_lstm})
# lstm_output.to_csv('E:/NLP/Kesci2019/Pre_competition/result/lstm_new.csv', index = False, quoting = 3)