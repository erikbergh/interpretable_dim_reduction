import tensorflow as tf
import numpy as np
import random


def gaussian(x, mu, sigma):
    sigma = tf.cast(sigma, tf.float32)
    sigma = tf.abs(sigma) #instead of square
    return tf.exp(tf.divide(tf.reduce_sum(-tf.square(x - mu), axis=-1), sigma))


def get_matrix_weights(x, mu, s):

    epsilon = 0.0000001
    x = tf.cast(x, tf.float32)
    mu = tf.cast(mu, tf.float32)
    x = tf.expand_dims(x, axis=1)

    gaussian_weights = gaussian(x, mu, s)
    
    gaussian_weights = gaussian_weights / (tf.reduce_sum(gaussian_weights, axis=[-1], keepdims=True) + epsilon)

    return gaussian_weights



class GaussianPositionsLayer(tf.keras.layers.Layer):
    def __init__(self, gaussian_points, dim1):
        super(GaussianPositionsLayer, self).__init__()

        self.gaussian_points = gaussian_points

        self.mu = tf.Variable(shape=([self.gaussian_points, dim1]), initial_value=np.random.random([gaussian_points, dim1]), trainable=True)





    def build(self, input_shape):
        pass
    def call(self, pos):
        mu = tf.cast(self.mu, tf.float32)
        return mu
    

class AssignLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, input_dim, n, gaussian_points):
        super(AssignLayer, self).__init__()
        self.output_dim = output_dim
        self.n = n
        self.gaussian_points = gaussian_points
 
        self.s = tf.Variable(shape=([self.gaussian_points]), initial_value=np.ones([self.gaussian_points]), trainable=True)
        self.w = self.add_weight(shape=(self.gaussian_points, input_dim, self.output_dim), initializer='random_normal', trainable=True)

    def build(self, input_shape):
        pass

    def call(self, pos, mu):
        matrix_weights = get_matrix_weights(pos, mu, self.s)
        w = tf.einsum('jkl,ij->ijkl', self.w, matrix_weights)

        w = tf.math.reduce_sum(w, axis=[1])

        output = tf.einsum('ij,ijk->ik', pos, w)
        return output


def createLoss(neighbours):
    class Loss(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            expanded_inputs = tf.expand_dims(y_pred, axis=1)
            expanded_outputs = tf.expand_dims(y_pred, axis=0)
            y_pred = tf.reduce_sum(tf.square(expanded_inputs - expanded_outputs), axis=-1)
            if neighbours != None:

                indices = tf.map_fn(lambda x: tf.cast(tf.nn.top_k(-x, k=neighbours+1, sorted=False).indices, tf.float32), tf.cast(y_pred, tf.float32))
                indices = tf.cast(indices, tf.int32)
                r = tf.shape(indices)[0]
                row_indices = tf.range(r)
                reshaped_row_indices = tf.reshape(row_indices, (-1, 1))
                tiled_row_indices = tf.tile(reshaped_row_indices, [1, neighbours+1])
                tiled_row_indices = tf.reshape(tiled_row_indices, (-1, 1))

                reshaped_input_matrix = tf.reshape(indices, [-1, 1])

                indices = tf.concat([tiled_row_indices, reshaped_input_matrix], axis=1)     
                y_true = tf.gather_nd(y_true, indices)
                y_pred = tf.gather_nd(y_pred, indices)
            dist = tf.square(y_true - y_pred)

            loss = tf.reduce_mean(dist)
            return loss
    return Loss()


def getDistances(vectors):
    distances = np.linalg.norm(vectors[:, np.newaxis] - vectors, axis=-1)
    return distances


class DimReduction():

    def __init__(self, gaussian_points=121, neighbours=None, dim2=2, train_gaussian_points=False):

        super(DimReduction, self).__init__()
        self.dim2 = dim2
        self.gaussian_points = gaussian_points
        self.neighbours = neighbours
        self.train_gaussian_points = train_gaussian_points

    def __setUpModels(self, vectors):
        
        self.n = vectors.shape[0]
        self.dim1 = vectors.shape[1]
        inputs_m = tf.keras.layers.Input(shape=(self.dim1,))
        inputs = tf.keras.layers.Input(shape=(self.dim1,))

        mu = GaussianPositionsLayer(self.gaussian_points, self.dim1)(inputs_m)
        self.mesh_model = tf.keras.Model(inputs=inputs_m, outputs=mu)
        mu2 = self.mesh_model(inputs)
        x = AssignLayer(self.dim2, self.dim1, self.n, self.gaussian_points)(inputs, mu2)
        self.model = tf.keras.Model(inputs=inputs, outputs=x)

        init = vectors + .00001 * (np.random.random(vectors.shape) - .5) / (np.max(vectors))
        self.model.layers[1].set_weights([np.array(random.sample(list(init), self.gaussian_points))])

    def fit(self, vectors, learning_rate=.01, max_epochs=1000, patience=None):
        
        self.distances = getDistances(vectors)
        self.__setUpModels(vectors)
        self.model.layers[1].trainable = self.train_gaussian_points
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=opt, loss=createLoss(self.neighbours))
        if patience == None:
            self.model.fit(vectors, self.distances, batch_size=self.n, epochs=max_epochs, shuffle=False)
        else:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
            self.model.fit(vectors, self.distances, batch_size=self.n, epochs=max_epochs, shuffle=False, callbacks=[early_stopping])


    def fit_transform(self, vectors, learning_rate=0.01, max_epochs=1000, patience=None):
       
        self.fit(vectors, learning_rate, max_epochs, patience)
        return self.transform(vectors)
        
    def transform(self, vectors):
        return self.model.predict(vectors, batch_size=vectors.shape[0])
    
    def get_gaussian_centers(self):
        return self.mesh_model.predict(np.zeros([1, self.dim1]))
    
    def get_dim_reduct_gaussian_centers(self):
        mu = self.get_gaussian_centers()
        matrix_weights = get_matrix_weights(mu, mu, self.model.layers[2].weights[0])
        w = tf.einsum('jkl,ij->ijkl', self.model.layers[2].weights[1], matrix_weights)
        w = tf.math.reduce_sum(w, axis=[1])

        return tf.einsum('ij,ijk->ik', mu, w).numpy()
    
    def __get_weights_mesh(self, l, ranges):

        if len(l) != self.dim2:
            raise Exception("The number of provided mesh points numbers (l) must equal the number of output dimension. In this case " + str(self.dim2) + ".") 
                
        if len(ranges) != self.dim2:
            raise Exception("The number of provided ranges must equal the number of output dimension. In this case " + str(self.dim2) + ".") 
        
        l = np.array(l)
        min = np.array(ranges)[:, 0]
        max = np.array(ranges)[:, 1]
        ranges = []
        for i in range(l.shape[0]):
            range_tensor = tf.linspace(min[i], max[i], l[i])
            ranges.append(range_tensor)

        grid = tf.meshgrid(*ranges, indexing='ij')
        meshgrid = tf.stack(grid, axis=-1)
        positions = tf.reshape(meshgrid, [-1, self.dim2])

        matrix_weights = get_matrix_weights(positions, self.get_dim_reduct_gaussian_centers(), self.model.layers[2].weights[0])
        w = tf.einsum('jkl,ij->ijkl', self.model.layers[2].weights[1], matrix_weights)
        w = tf.math.reduce_sum(w, axis=[1])
        return w.numpy(), positions
    
    def get_influence_mesh(self, l, ranges):
        if len(l) != self.dim2:
            raise Exception("The number of provided mesh points numbers (l) must equal the number of output dimension. In this case " + str(self.dim2) + ".") 
                
        if len(ranges) != self.dim2:
            raise Exception("The number of provided ranges must equal the number of output dimension. In this case " + str(self.dim2) + ".") 
        w, positions = self.__get_weights_mesh(l, ranges)
        w = np.abs(w)
        w = tf.math.reduce_sum(w, axis=[-1], keepdims=True) / tf.math.reduce_sum(w, axis=[-1, -2],keepdims=True)
        return tf.math.reduce_sum(w, axis=[-1]).numpy(), positions
    
    def get_weights_mesh(self, l, ranges):
        if len(l) != self.dim2:
            raise Exception("The number of provided mesh points numbers (l) must equal the number of output dimension. In this case " + str(self.dim2) + ".") 
                
        if len(ranges) != self.dim2:
            raise Exception("The number of provided ranges must equal the number of output dimension. In this case " + str(self.dim2) + ".") 
 
        return self.__get_weights_mesh(l, ranges)
    
    def get_matrix_norm_mesh(self, l, ranges):
        if len(l) != self.dim2:
            raise Exception("The number of provided mesh points numbers (l) must equal the number of output dimension. In this case " + str(self.dim2) + ".") 
                
        if len(ranges) != self.dim2:
            raise Exception("The number of provided ranges must equal the number of output dimension. In this case " + str(self.dim2) + ".") 
        w, positions = self.__get_weights_mesh(l, ranges)
        return tf.linalg.norm(w, axis=[-1, -2]).numpy(), positions.numpy()
    
    def get_component_variation_mesh(self, l, ranges):
        if len(l) != self.dim2:
            raise Exception("The number of provided mesh points numbers (l) must equal the number of output dimension. In this case " + str(self.dim2) + ".") 
                
        if len(ranges) != self.dim2:
            raise Exception("The number of provided ranges must equal the number of output dimension. In this case " + str(self.dim2) + ".") 
        w, positions = self.__get_weights_mesh(l, ranges)
        w = np.abs(w)
        w = tf.math.reduce_sum(w, axis=[-1], keepdims=True) / tf.math.reduce_sum(w, axis=[-1, -2],keepdims=True)
        w = tf.math.reduce_sum(w, axis=[-1])
        return tf.math.reduce_variance(w, axis=-1).numpy(), positions
        
    def get_transformations(self):
        return self.model.layers[2].weights[1].numpy()
    
    def get_sigmas(self):
        return self.model.layers[2].weights[0].numpy()
    
    def get_mean_transformation(self):
        w = self.model.layers[2].weights[1]
        w = tf.math.reduce_mean(w, axis=[0])
        return w.numpy()
    
    def get_dimension_influence(self):
        w = self.model.layers[2].weights[1]
        w = np.abs(w)
        w = tf.math.reduce_sum(w, axis=[-1], keepdims=True) / tf.math.reduce_sum(w, axis=[-1, -2], keepdims=True)
        w = tf.math.reduce_mean(w, axis=[0])
        return w.numpy()
    
    def get_reconstruction_error(self, vectors):
        d1 = getDistances(vectors)
        delta = tf.reduce_sum(tf.abs(d1 - getDistances(self.model.predict(vectors))))
        error = delta / tf.reduce_sum(d1)
        return error.numpy()