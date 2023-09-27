The script generates data for multiclassification problem and then predicts outcomes. 

Steps for data generation:
- generate random coefficients according to a uniform distribution from -1 to 1 for 50123 conical sections.
- select some coefficients randomly and set them to zero. So there are fewer hyperbolas and ellipses.
- calculate the determinants according to the article https://en.wikipedia.org/wiki/Degenerate_conic#Discriminant  
- identify and filter out degenerate conic sections.
- calculate discriminants and determine the types of conic sections.
- remove the calculated discriminants to simulate a lack of knowledge in the subject area.
- divide the generated data into training and test samples. For the test sample we use 20% of the original data.
 
Docker is required.
Uncomment two lines in order to use GPU for model training.
