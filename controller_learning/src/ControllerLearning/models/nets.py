"""
每次更改网络直接复制
当前为: nets_tcn_tf.py
"""
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Conv1D, ReLU, Layer, Activation
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D

try:
    from .tf_addons_normalizations import InstanceNormalization
except:
    from tf_addons_normalizations import InstanceNormalization

def create_network(settings):
    net = AggressiveNet(settings)
    return net


class Network(Model):
    def __init__(self):
        super(Network, self).__init__()

    def create(self):
        self._create()

    def call(self, x):
        return self._internal_call(x)

    def _create(self):
        raise NotImplementedError

    def _internal_call(self):
        raise NotImplementedError

# TCN ---
class Chomp1d(Layer):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def call(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(Layer):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, activation='relu'):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv1D(n_outputs, kernel_size, strides=stride, padding='causal', dilation_rate=dilation)
        self.chomp1 = Chomp1d(padding)
        self.activation1 = Activation(activation)
        self.dropout1 = tf.keras.layers.SpatialDropout1D(dropout)

        self.conv2 = Conv1D(n_outputs, kernel_size, strides=stride, padding='causal', dilation_rate=dilation)
        self.chomp2 = Chomp1d(padding)
        self.activation2 = Activation(activation)
        self.dropout2 = tf.keras.layers.SpatialDropout1D(dropout)

        self.downsample = Conv1D(n_outputs, 1, padding='same') if n_inputs != n_outputs else lambda x: x
        # if n_inputs == n_outputs:
        #     self.downsample = lambda x: x[:, :, :y.shape[2]]

    def call(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.activation1(y)
        y = self.dropout1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.activation2(y)
        y = self.dropout2(y)

        res = self.downsample(x)
        # res = res[:, :, :y.shape[2]]  # 调整res的时间维度以匹配y
        if res.shape[2] != y.shape[2]:
            res = res[:, :, :y.shape[2]]  # 在运行时调整res的时间维度以匹配y
        # print("y shape:", y.shape)
        # print("res shape:", res.shape)
        return tf.keras.activations.relu(y + res)

class TemporalConvNet(Layer):
    def __init__(self, num_inputs, num_hidden_channels, kernel_size=2, dropout=0.2, activation='relu'):
        super(TemporalConvNet, self).__init__()
        self.layers = []
        num_levels = len(num_hidden_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_hidden_channels[i - 1]
            out_channels = num_hidden_channels[i]
            self.layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size, dropout=dropout, activation=activation
            ))

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# TCN ---

class AggressiveNet(Network):
    def __init__(self, config):
        super(AggressiveNet, self).__init__()
        self.config = config
        self._create(input_size=(config.seq_len, config.min_number_fts, 5))

    def _create(self, input_size, has_bias=True, learn_affine=True):
        """Init.
        Args:
            input_size (float): size of input
            has_bias (bool, optional): Defaults to True. Conv1d bias?
            learn_affine (bool, optional): Defaults to True. InstanceNorm1d affine?
        """
        # activation layers option
        GELU =  tf.keras.layers.Activation('gelu')
        LReLU = LeakyReLU(alpha=1e-2)
        dict_activation = {"ReLU": ReLU(), "GELU": GELU, "LeakyReLU": LReLU}
        # activation = dict_activation['LeakyReLU']
        activation = LeakyReLU(alpha=1e-2)
        if self.config.use_fts_tracks:
            f = 2.0
            # dilation_rate=1时采用普通卷积，dilation_rate=2时采用空洞卷积。
            # use_bias 配置该层的神经网络是否使用偏置向量
            # pointnet是处理图像中特征点的神经网络
            print(f"pointnet use LeakyReLU(alpha=1e-2)")
            self.pointnet = [Conv2D(int(16 * f), kernel_size=1, strides=1, padding='valid',
                                    dilation_rate=1, use_bias=has_bias, input_shape=input_size),
                             InstanceNormalization(axis=3, epsilon=1e-5, center=learn_affine, scale=learn_affine),
                             activation,
                             Conv2D(int(32 * f), kernel_size=1, strides=1, padding='valid', dilation_rate=1,
                                    use_bias=has_bias),
                             InstanceNormalization(axis=3, epsilon=1e-5, center=learn_affine, scale=learn_affine),
                             activation,
                             Conv2D(int(64 * f), kernel_size=1, strides=1, padding='valid', dilation_rate=1,
                                    use_bias=has_bias),
                             InstanceNormalization(axis=3, epsilon=1e-5, center=learn_affine, scale=learn_affine),
                             activation,
                             Conv2D(int(64 * f), kernel_size=1, strides=1, padding='valid', dilation_rate=1,
                                    use_bias=has_bias),
                             GlobalAveragePooling2D()]

            # fts_mergenet是特征轨迹
            input_size = (self.config.seq_len, int(64*f))
            activation = dict_activation['GELU']
            # 将 TCN 架构与其他必要层整合到列表中
            self.fts_mergenet = [
                TemporalConvNet(
                num_inputs=input_size,  # 输入通道数
                num_hidden_channels=[int(64 * f), int(32 * f), int(32 * f), int(32 * f)],  # 隐藏层通道数
                kernel_size=2,  # 卷积核大小
                dropout=0.2,  # Dropout 比率
                activation=activation  # 激活函数
                ),
                Flatten(),
                Dense(int(64 * f))
            ]

        # states_conv是融合imu
        g = 2.0
        
        self.states_conv = [
                TemporalConvNet(
                num_inputs=input_size,  # 输入通道数
                num_hidden_channels=[int(64 * g), int(32 * g), int(32 * g), int(32 * g)],  # 隐藏层通道数
                kernel_size=2,  # 卷积核大小
                dropout=0.2,  # Dropout 比率
                activation=activation  # 激活函数
                ),
                Flatten(),
                Dense(int(64 * g))
            ]

        # 这是MLP网络
        self.control_module = [Dense(64*g),
                               activation,
                               Dense(32*g),
                               activation,
                               Dense(16*g),
                               activation,
                               Dense(4)]

        # # 这里换成一维卷积
        activation = LeakyReLU(alpha=1e-2)
        # # input_shape = (self.config.batch_size, self.config.seq_len, 256)
        # inputs = tf.keras.Input(shape=(1, 256))
        self.conv1d_control_module = [
            tf.keras.layers.Conv1D(filters=128, kernel_size=1, padding='same', activation=activation),
            tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding='same', activation=activation),
            tf.keras.layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=activation),
            tf.keras.layers.Conv1D(filters=4, kernel_size=1, padding='same', activation=activation)
        ]

    def _pointnet_branch(self, single_t_features):
        x = tf.expand_dims(single_t_features, axis=1)
        for f in self.pointnet:
            x = f(x)
        return x

    def _features_branch(self, input_features):
        preprocessed_fts = tf.map_fn(self._pointnet_branch,
                                     elems=input_features,
                                     parallel_iterations=self.config.seq_len) # (seq_len, batch_size, 64)
        preprocessed_fts = tf.transpose(preprocessed_fts, (1,0,2)) # (batch_size, seq_len, 64)
        x = preprocessed_fts
        for f in self.fts_mergenet:
            x = f(x)
        return x

    def _states_branch(self, embeddings):
        x = embeddings
        for f in self.states_conv:
            x = f(x)
        return x

    def _control_branch(self, embeddings):
        mode = 'conv1d'
        if mode == 'conv1d':
            embeddings = tf.expand_dims(embeddings, axis=1)
            # 调整 embeddings 的形状，确保它是 (batch_size, 1, 256)
            # x = tf.reshape(embeddings, (-1, 1, 256)) # (batch_size, 1, 256)
            x = embeddings
            # print(f'control_branch input shape: {x.shape}')
            
            # activation = LeakyReLU(alpha=1e-2)

            # # 创建卷积层模型
            # inputs = tf.keras.Input(shape=(1, 256))
            # x = tf.keras.layers.Conv1D(filters=128, kernel_size=1, padding='same', activation=activation)(inputs)
            # x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding='same', activation=activation)(x)
            # x = tf.keras.layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=activation)(x)
            # x = tf.keras.layers.Conv1D(filters=4, kernel_size=1, padding='same', activation=activation)(x)

            # model = tf.keras.Model(inputs=inputs, outputs=x)

            # # 应用模型
            # output = model(embeddings)  # 确保此时的输入 shape 正确
            # output = tf.squeeze(output, axis=1)
            for f in self.conv1d_control_module:
                x = f(x)
            x = tf.squeeze(x, axis=1)
            return x
        elif mode == 'dense':
            x = embeddings
            for f in self.control_module:
                x = f(x)
            return x

    def _internal_call(self, inputs):
        # 这里是融合的地方
        states = inputs['state']
        # 如果使用特征轨迹
        if self.config.use_fts_tracks:
            fts_stack = inputs['fts']  # (batch_size, seq_len, min_numb_features, 5)
            fts_stack = tf.transpose(fts_stack, (1,0,2,3)) # (seq_len, batch_size, min_numb_features, 5)
            # Execute PointNet Part
            fts_embeddings = self._features_branch(fts_stack) # (32, 128)
            # print(f'=====fts_embeddings shape: {fts_embeddings.shape}=====')
        states_embeddings = self._states_branch(states) # (32, 128)
        # print(f'=====states_embeddings shape: {states_embeddings.shape}=====')
        if self.config.use_fts_tracks:
            total_embeddings = tf.concat((fts_embeddings, states_embeddings), axis=1) #(32, 256)
        else:
            total_embeddings = states_embeddings
        # print(f'=====total_embeddings shape: {total_embeddings.shape}=====')
        
        # 将 total_embeddings 形状从 (batch_size, 256) 转换为 (batch_size, 1, 256)
        # total_embeddings = tf.expand_dims(total_embeddings, axis=1)
        # print(f'total_embeddings shape: {total_embeddings.shape}')
        output = self._control_branch(total_embeddings)
        # print(f'======================output shape: {output.shape}======================')
        return output