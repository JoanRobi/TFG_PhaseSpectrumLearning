## Model de referència utilitzat a la comparació



Per a la comparació experimental amb un model de referència s'ha emprat FFD

(Facial Forgery Detection), corresponent al treball:



@inproceedings{cvpr2020-dang,

  title={On the Detection of Digital Face Manipulation},

  author={Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, Anil Jain},

  booktitle={In Proceeding of IEEE Computer Vision and Pattern Recognition (CVPR 2020)},

  address={Seattle, WA},

  year={2020}

}



El codi original de FFD no es redistribueix dins aquest repositori.

Per reproduir la comparació, s'ha de descarregar el repositori oficial de FFD

i preparar-ne l'estructura de dades mitjançant l'script propi

`prepara_ffd_faceapp.py`.



Aquest repositori només inclou el codi desenvolupat per al model

PhaseSpectrumLearning i els scripts auxiliars necessaris per preparar

l'experiment comparatiu.



El link al repoditori de FFD es el següent: https://github.com/JStehouwer/FFD_CVPR2020