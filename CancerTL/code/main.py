import os
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical
from CancerTL import *
from MLP_1v1 import *
from MLP_1v1_3layer import *
from MLP_multiclass_model import *
from MLP_multiclass_model_3layer import *
from RF_model_baseline_10_fold import *
from GB_baseline_10_fold import *
from AdaBoost_baseline_10_fold import *
from LR_baseline_10_fold import *
from SVM_baseline_10_fold import *
import yaml
import sys


#CONFIG_PATH = "./config/"

CONFIG_PATH = "../results/version6/config/"

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

#save log
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



config = load_config("Mainconfig.yaml")


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# set seed
os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed(config["genral_seed"])

# data preprocess
df = pd.read_csv("../data/Student_DifferentCancerDetection.csv")
X = df.drop(['Sex', 'Tumor type', 'AJCC Stage'], axis=1)  ## gender/Sex
y = df['Tumor type']
y = np.array(y)
Xcolname = X.columns.values.tolist()

# labelencoder
le = LabelEncoder()
le.fit(y)
print(list(le.classes_))
y = le.transform(y)
y = y.astype(np.int32)
X = np.array(X)
# onehot encoder
yb = to_categorical(y)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=config["StratifiedFold_seed"])

version = config["version"]

logfile = 'test'
sys.stdout = Logger('../results/' + version + '/' + version + '_' +logfile + '.log', sys.stdout)
sys.stderr = Logger('../results/' + version + '/' + version + '_' +logfile + '.log_file', sys.stderr)

# %%
for i, (trainvalid_index, test_index) in enumerate(cv.split(X, y)):
    print("TRAIN:", trainvalid_index, "TEST:", test_index)

    prefix = '../results/' + version + '/fold' + str(i)

    print("This is fold", i)

    trainvalidX, testX = X[trainvalid_index], X[test_index]
    trainvalidY, testY = y[trainvalid_index], y[test_index]
    trainvalidYb, testYb = yb[trainvalid_index], yb[test_index]
    trainX, validX, trainY, validY, trainYb, validYb = model_selection.train_test_split(trainvalidX, trainvalidY, trainvalidYb, stratify=trainvalidY, train_size=0.90, random_state=config["TrainValidSplit_seed"])  # 10fold

    # standarlization
    Scaler = StandardScaler()
    Scaler.fit(trainX)
    trainXn = Scaler.transform(trainX)
    validXn = Scaler.transform(validX)
    testXn = Scaler.transform(testX)

    # Imbalanced data resampling 20210714
    trainXn_UP, trainY_UP = Upsampling_SMOTE(trainXn, trainY)
    trainYb_UP = to_categorical(trainY_UP)

    # main
    CancerTL(prefix, trainXn_UP, trainY_UP, trainYb_UP, validXn, validY, validYb, testXn, testY, testYb, Xcolname)

