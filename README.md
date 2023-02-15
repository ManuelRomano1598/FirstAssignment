# FirstAssignment

First Assignment for the NLP exam 


Initially, I had a bit of difficulty choosing the right corpus. I settled for the europarl_raw because it's easly divisible by language. 

This gave me a large number of documents I could use as the "non-English" class and just one for the "English", 

a good addition to the model would be adding other English corpses with different dialects and slangs. 

I first tried a cap on the non-English to help with the slow performance by limiting the number of chapters per language.

I later chose to remove the cap and decrease the number by avoiding similar languages (like Spanish and Portuguese).


After playing with the number of the possible size differences between training and test data I opted for an 80% - 20% split.


I used a library to create the confusion matrix from the list of guesses and the list of golden labels, using the matrix I then

calculated precision, recall and accuracy.

As the theory suggested i focused on recall and precision to judge my algorthim, i was worried about the perfect score i was getting 

so i spent a long time testing.

I think that the corpus i used, being a transcription of a European congress conference, isn't similar enough to trick the algorithm. 

After feeding, as test data, some mixed and wrong sentences I came to the conclusion that with a different corpus (with more slangs and 

Foreign words) more realistic evaluation metrics could be achieved.

Another factor may be the decision of using chapters as documents. The precision on a lower level document (like sentences) may have been 

lower since sporadic ambiguous words would have had more impact.
