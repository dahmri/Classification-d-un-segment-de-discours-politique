from preprocessing import load_data
import itertools
import pandas as pd
import collections
import numpy as np
from gensim.models import word2vec, KeyedVectors
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding,GlobalMaxPooling1D,LSTM, MaxPooling1D
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, classification_report,roc_curve
np.random.seed(0)



def get_word2vec(vocabulary):
    """
    Getting W2V vectors
    """
    model_name="model.txt"
    embedding_model = KeyedVectors.load_word2vec_format(model_name,unicode_errors='ignore')
    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
                              np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for word, key in vocabulary.items()}
    return embedding_weights



def splitting_Data(sentences,length):
    """
    Splitting Data for train and test
    Taking 2000 sentences for test (1000 from each class)
    """

    X_test_class0=sentences[:1000]
    X_train_class0=sentences[1000:length]

    X_test_class1=sentences[length:length+1000]
    X_train_class1=sentences[length+1000:]

    X_train=np.concatenate((X_train_class0,X_train_class1))
    X_test=np.concatenate((X_test_class0,X_test_class1))

    Y_test_class0=labels[:1000]
    Y_train_class0=labels[1000:length]

    Y_test_class1=labels[length:length+1000]
    Y_train_class1=labels[length+1000:]

    Y_train=np.concatenate((Y_train_class0,Y_train_class1))
    Y_test=np.concatenate((Y_test_class0,Y_test_class1))

    return X_train, Y_train, X_test, Y_test


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    
    sentences, labels , vocabulary = load_data()

    #Ploting number of texts for the two classes
    counter=collections.Counter(labels)
    key, value =zip(*counter.items())
    indexes = np.arange(len(key))
    plt.bar(indexes, value,0.8,color="blue")
    plt.xticks(indexes, ['Chirac','Mitterrand'])
    plt.xlabel("Classe")
    plt.ylabel("Nombre de textes")
    plt.text(0, value[0]+200, str(round((value[0]/(value[0]+value[1]))*100))+"%" , fontweight='bold',horizontalalignment='center')
    plt.text(1, value[1]+200, str(round((value[1]/(value[0]+value[1]))*100))+"%" , fontweight='bold',horizontalalignment='center')
    plt.show()


    print ("Splitting the data...")
    X_train, Y_train, X_test, Y_test = splitting_Data(sentences,value[0])




    shuffle_indices = np.random.permutation(np.arange(len(X_train)))
    X_train = X_train[shuffle_indices]
    Y_train = Y_train[shuffle_indices]
    sequence_length=len(X_train[0])

    print ("Loading Word2vec....")
    W2V_weights=np.loadtxt('./Data/W2V_weights', delimiter=',')
    print ("Loading has finished")


    model = Sequential()
    model.add(Embedding(len(vocabulary), 100, input_length=sequence_length, name="embedding"))
    model.add(Convolution1D(filters=100,kernel_size=3,padding="valid", activation="relu",strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    ES = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    CP = ModelCheckpoint(filepath="./Model/best_model.hdf5", verbose=0, save_best_only=True)

    model.summary()

    print("Initializing embedding layer with word2vec weights, shape", W2V_weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([W2V_weights])

    print ("Starting training....")
    # Train the model
    history=model.fit(X_train, Y_train, batch_size=64,callbacks=[ES,CP], epochs=100, validation_split=0.2, verbose=2)


    history_dict = history.history

    #Ploting training loss and validation loss
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values)+1)
    plt.plot(epochs, loss_values, 'r', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    #Ploting training accuracy and validation accuracy
    plt.clf()
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plt.plot(epochs, acc_values, 'r', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    #Loading the best Model for testing
    model.load_weights('./Model/best_model.hdf5')

    #Testing the model
    Y_pred=model.predict(X_test)
    y_classes = model.predict_classes(X_test)
    Y_test = list(map(int, Y_test))
    Y_pred_mitterrand=Y_pred[-1000:]
    Y_pred_chirac=Y_pred[:1000]


    #Ploting the distribution of the prediction
    plt.hist(Y_pred_chirac,bins=100,label="Chirac",color="b")
    plt.hist(Y_pred_mitterrand,bins=100,label="Mitterrand",color="r")
    plt.xlabel("Probabilités")
    plt.ylabel("Nombre de prédictions")
    plt.title('Distribution de la prédiction')
    plt.legend()
    plt.show()

    #Ploting confusion_matrix
    CM=confusion_matrix(Y_test, y_classes)
    plot_confusion_matrix(CM, classes=["Chirac", "Mitterrand"], title='Matrice de confusion')
    classification = classification_report(Y_test, y_classes)
    print (classification)


    #Computing the roc_auc
    roc_auc=roc_auc_score(Y_test, Y_pred)
    print (roc_auc)

    fp, tp, thresholds_keras= roc_curve(Y_test, Y_pred)
    #Ploting the ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fp, tp, label='(area = {:.2f})'.format(roc_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
