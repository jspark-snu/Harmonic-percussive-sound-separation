
import os
import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt




def NMF(X,Kh=500,Kp=250,max_iter=100):
	Nf = X.shape[0]
	Nt = X.shape[1]
	W_esti = np.random.rand(Nf,Kh+Kp)
	H_esti = np.random.rand(Kh+Kp,Nt)

	# Flat initialization
	W_esti[:,Kh:Kh+Kp]=np.ones((Nf,Kp))

	for i in range(0,max_iter):
		H_esti = H_esti*np.matmul(W_esti.T,X/np.matmul(W_esti,H_esti))
		Denominator = np.matmul(np.sum(W_esti,axis=0)[:,None],np.ones((1,Nt)))
		H_esti = H_esti/Denominator

		# Dirichlet prior (generalized to NMF framework)
		if i<(max_iter-1):
			# Harmonic
			neighbors = np.concatenate((H_esti[0:Kh,1:Nt],H_esti[0:Kh,Nt-1:Nt]),axis=1)
			delta = H_esti[0:Kh,:]-neighbors
			delta = 0.7*delta
			H_esti[0:Kh,:] = neighbors+delta
			H_esti[0:Kh,-1] = neighbors[:,-1]

			# Percussive
			neighbors = np.concatenate((H_esti[Kh:Kh+Kp,1:Nt],H_esti[Kh:Kh+Kp,Nt-1:Nt]),axis=1)
			delta = H_esti[Kh:Kh+Kp,:]-neighbors
			delta = 1.05*delta
			H_esti[Kh:Kh+Kp,:] = np.maximum(neighbors+delta,0.00000001)
			H_esti[Kh:Kh+Kp,-1] = neighbors[:,-1]

		W_esti = W_esti*np.matmul(X/np.matmul(W_esti,H_esti),H_esti.T)
		Denominator2 = np.matmul(np.ones((Nf,1)),np.sum(H_esti,axis=1)[None,:])
		W_esti = W_esti/Denominator2

		# Dirichlet prior (generalized to NMF framework)
		if i<(max_iter-1):
			# Harmonic
			neighbors = np.concatenate((W_esti[1:Nf,0:Kh],W_esti[Nf-1:Nf,0:Kh]),axis=0)
			delta = W_esti[:,0:Kh]-neighbors
			delta = 1.05*delta
			W_esti[:,0:Kh] = np.maximum(neighbors+delta,0.00000001)
			W_esti[-1,0:Kh] = neighbors[-1,:]

			# Percussive
			neighbors = np.concatenate((W_esti[1:Nf,Kh:Kh+Kp],W_esti[Nf-1:Nf,Kh:Kh+Kp]),axis=0)
			delta = W_esti[:,Kh:Kh+Kp]-neighbors
			delta = 0.95*delta
			W_esti[:,Kh:Kh+Kp] = neighbors+delta
			W_esti[-1,Kh:Kh+Kp] = neighbors[-1,:]

	return (W_esti,H_esti)



def HPSS(x,sr=44100,wiener_filt=0):
	# Audio signals are assumed to be mono
	X = librosa.core.stft(x,n_fft=4096,hop_length=1024,window='hamming')
	X_abs = np.abs(X)+0.0000000001
	phase = X/X_abs

	Kh = 500
	Kp = 250
	max_iter = 100
	W_esti,H_esti = NMF(X_abs,Kh,Kp,max_iter)

	# Post-processing
	X_harmonic = np.matmul(W_esti[:,0:Kh],H_esti[0:Kh,:])
	X_percussive = np.matmul(W_esti[:,Kh:Kh+Kp],H_esti[Kh:Kh+Kp,:])

	if (wiener_filt==1):
		X_harmonic2 = np.square(X_harmonic)/(np.square(X_harmonic)+np.square(X_percussive))*X_abs
		X_percussive = np.square(X_percussive)/(np.square(X_harmonic)+np.square(X_percussive))*X_abs
		X_harmonic = X_harmonic2


	harmonic = librosa.core.istft(X_harmonic*phase,hop_length=1024,window='hamming')
	percussive = librosa.core.istft(X_percussive*phase,hop_length=1024,window='hamming')

	return (harmonic,percussive)


def main():
	### Example
	filename = './example.mp3'
	x, fs = librosa.core.load(filename,sr=44100,mono=True)

	harmonic,percussive = HPSS(x,sr=44100,wiener_filt=1)

	# Save
	librosa.output.write_wav(os.path.splitext(filename)[0]+'_H.wav',harmonic,fs)
	librosa.output.write_wav(os.path.splitext(filename)[0]+'_P.wav',percussive,fs)


def main_toy():
	X1 = np.zeros((100,100))
	X2 = np.zeros((100,100))

	X1[20,:]=1
	X1[40,:]=1
	X1[80,:]=1
	X2[:,30]=1
	X2[:,80]=1
	W_esti,H_esti = NMF(X1+X2,Kh=500,Kp=250,max_iter=100)

	X_harmonic = np.matmul(W_esti[:,0:500],H_esti[0:500,:])
	X_percussive = np.matmul(W_esti[:,500:750],H_esti[500:750,:])

	fig = plt.figure()
	sp1 = fig.add_subplot(1,2,1)
	sp1.imshow(X_harmonic)

	sp2 = fig.add_subplot(1,2,2)
	sp2.imshow(X_percussive)
	plt.show()



if __name__ == "__main__":
	main()


