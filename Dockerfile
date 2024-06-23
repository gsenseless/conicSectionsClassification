FROM jupyter/datascience-notebook

#COPY conicSectionsClassification.py /home/jovyan/work/conicSectionsClassification.py

RUN pip install autogluon
CMD python /home/jovyan/work/conicSectionsClassification.py --approach automl

#CMD python /home/jovyan/work/conicSectionsClassification.py --approach classic

