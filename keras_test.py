import keras
from keras import losses

losses.me

def negative_log_likelihood(ytime, ystatus):
    LL_i = T.switch(T.eq(ystatus[i],1), self.theta - T.log(T.sum(self.exp_theta * T.gt(ytime, ytime[i]))),0)

def get_negative_log_likelihood(self, y_true, X, mask):
    """Compute the loss, i.e., negative log likelihood (normalize by number of time steps)
       likelihood = 1/Z * exp(-E) ->  neg_log_like = - log(1/Z * exp(-E)) = logZ + E
    """
    input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
    if self.use_boundary:
        input_energy = self.add_boundary_energy(input_energy, mask, self.left_boundary, self.right_boundary)
    energy = self.get_energy(y_true, input_energy, mask)
    logZ = self.get_log_normalization_constant(input_energy, mask, input_length=K.int_shape(X)[1])
    nloglik = logZ + energy
    if mask is not None:
        nloglik = nloglik / K.sum(K.cast(mask, K.floatx()), 1)
    else:
        nloglik = nloglik / K.cast(K.shape(X)[1], K.floatx())
    return nloglik