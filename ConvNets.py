from itertools import permutations
import tensorflow as tf
import random
import numpy as np
import pdb

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# tf.Session(config=config)

# from keras.backend.tensorflow_backend import set_session
# set_session(tf.Session(config=config))


class DeepNN():
    def __init__(self, args, stepSize):
        # game params
        self.N = args.N
        self.K = args.K
        self.M = args.M
        self.Q = args.Q
        self.stepSize = stepSize
        self.numfilters1 = args.numfilters1
        self.numfilters2 = args.numfilters2
        self.batchSize = args.batchSize
        self.l2_const = args.l2_const

        self.numMoves = self.Q ** self.stepSize

        # learning rate
        self.lr = args.lr  # Ensure `args` has an `lr` attribute

        # build DNN
        self.build_DNN()

        # save and load Params
        self.cost_his = []

        # policy entropy & cross entropy
        self.PolicyEntropy = 0
        self.CrossEntropy = 0
        self.cntEntropy = 0

    def build_DNN(self):
        # Neural Net
        # Define inputs
        self.batchInput = tf.keras.Input(shape=(self.M, self.N, self.K), dtype=tf.float32)
        self.dropRate = tf.keras.Input(shape=(), dtype=tf.float32)  # Dropout rate as input

        x_img = tf.reshape(self.batchInput, [-1, self.N, self.K, self.M])

        # Conv0
        conv0 = tf.keras.layers.Conv2D(self.numfilters1, kernel_size=[3, 3], padding='same')(x_img)
        conv0 = tf.keras.layers.BatchNormalization(axis=-1)(conv0)
        conv0 = tf.keras.layers.ReLU()(conv0)

        # Conv1
        conv1 = tf.keras.layers.Conv2D(self.numfilters1, kernel_size=[3, 3], padding='same')(conv0)
        conv1 = tf.keras.layers.BatchNormalization(axis=-1)(conv1)
        conv1 = tf.keras.layers.ReLU()(conv1)

        # Conv2
        conv2 = tf.keras.layers.Conv2D(self.numfilters1, kernel_size=[3, 3], padding='same')(conv1)
        conv2 = tf.keras.layers.BatchNormalization(axis=-1)(conv2)
        conv2 = tf.keras.layers.ReLU()(conv2)

        # Conv3
        conv3 = tf.keras.layers.Conv2D(self.numfilters1, kernel_size=[3, 3], padding='same')(conv2)
        conv3 = tf.keras.layers.BatchNormalization(axis=-1)(conv3)
        conv3 = tf.keras.layers.ReLU()(conv3)

        # Output PiVec
        x4 = tf.keras.layers.Conv2D(2, kernel_size=[1, 1], padding='same')(conv3)
        x4 = tf.keras.layers.BatchNormalization(axis=-1)(x4)
        x4 = tf.keras.layers.ReLU()(x4)

        x4_flat = tf.reshape(x4, [-1, 2 * (self.N) * (self.K)])

        x5 = tf.keras.layers.Dense(self.numfilters2)(x4_flat)
        x5 = tf.keras.layers.BatchNormalization(axis=1)(x5)
        x5 = tf.keras.layers.ReLU()(x5)
        x5_drop = tf.keras.layers.Dropout(rate=self.dropRate)(x5)

        self.piVec = tf.keras.layers.Softmax()(tf.keras.layers.Dense(self.numMoves)(x5_drop))

        # Output zValue
        y4 = tf.keras.layers.Conv2D(1, kernel_size=[1, 1], padding='same')(conv3)
        y4 = tf.keras.layers.BatchNormalization(axis=-1)(y4)
        y4 = tf.keras.layers.ReLU()(y4)

        y4_flat = tf.reshape(y4, [-1, 1 * (self.N) * (self.K)])

        y5 = tf.keras.layers.Dense(int(self.numfilters2 / 2))(y4_flat)
        y5 = tf.keras.layers.BatchNormalization(axis=1)(y5)
        y5 = tf.keras.layers.ReLU()(y5)
        y5_drop = tf.keras.layers.Dropout(rate=self.dropRate)(y5)

        self.zValue = tf.keras.layers.Dense(1, activation='tanh')(y5_drop)

        # Define the model
        self.model = tf.keras.Model(inputs=[self.batchInput, self.dropRate], outputs=[self.piVec, self.zValue])

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss={
                "piVec": tf.keras.losses.CategoricalCrossentropy(),
                "zValue": tf.keras.losses.MeanSquaredError(),
            },
        )

    def residual_block(self, input_layer, output_channel):

        conv1 = tf.layers.batch_normalization(input_layer, axis=-1, training=self.isTraining)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.conv2d(conv1, output_channel, kernel_size=[3,3], padding='same')

        
        conv2 = tf.layers.batch_normalization(conv1, axis=-1, training=self.isTraining)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.conv2d(conv2, output_channel, kernel_size=[3,3], padding='same')

        output = conv2 + input_layer
        return output

    def refresh_entropy(self):
        self.PolicyEntropy = 0
        self.CrossEntropy = 0
        self.cntEntropy = 0

    def output_entropy(self):
        return self.PolicyEntropy, self.CrossEntropy, self.cntEntropy

    def evaluate_node(self, rawstate, selfplay):
        # Extract features from the raw state
        state = self.feature_extract(rawstate)

        # Use the model to predict piVec and zValue
        piVec, zValue = self.model.predict([state, np.zeros((state.shape[0],))], verbose=0)

        return piVec, zValue
    
    def update_DNN(self, mini_batch, lr):
        # Expand mini_batch
        state_batch = np.array([data[0] for data in mini_batch])
        piVec_batch = np.array([data[1] for data in mini_batch])
        reward_batch = np.array([data[2] for data in mini_batch])[:, np.newaxis]

        # Extract features from the state batch
        state_batch = self.feature_extract(state_batch)

        # Train the model
        history = self.model.fit(
            x={"batchInput": state_batch, "dropRate": np.full((state_batch.shape[0],), 0.3)},
            y={"piVec": piVec_batch, "zValue": reward_batch},
            batch_size=self.batchSize,
            epochs=1,
            verbose=0,
        )

        # Append the loss to the cost history
        self.cost_his.append(history.history["loss"])

    def feature_extract(self,state_batch):
        feature1 = np.copy(state_batch)
        feature1[feature1 == -1] = 0
        feature2 = np.copy(state_batch)
        feature2[feature2 == 1] = 0
        feature2[feature2 == -1] = 1
        feature3 = np.zeros(state_batch.shape)
        feature3[state_batch == 0] = 1

        state = np.reshape(np.hstack((feature1,feature2,feature3)),(len(feature1),self.M, self.N, self.K))
        return state

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def saveParams(self, path):
        self.model.save_weights(path)

    def loadParams(self, path):
        self.model.load_weights(path)

    def get_params(self):
        return self.graph.get_collection('variables'), self.sess.run(self.graph.get_collection('variables'))
