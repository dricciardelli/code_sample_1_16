This directory contains images scaped from hotornot.com in early 2003.

Elements in MATLAB file:

rectimgs           1xN                 cell                

  The images scraped from hotornot.com.  Specifically, this is a cell
  array of 86x86x3 faces that have been extracted using a face detector
  and rectified.

score              1xN                 double   

  This is the corresponding score for that face, as determined by 
  users of the hotornot website.

votes              1xN                 double 

  This is the number of votes received for each face.  Higher votes
  tends to indicate a more accurate score.

rect_score         Nx5                 double

  These are the score for the rectification process.  There are
  five features used (right eye, left eye, nose, right corner of mouth,
  left corner of mouth).  I can't remember if higher is a better match
  or lower.  See Berg et al "Names and Faces in the News" CVPR 2004
  for more information about the rectification process.


If you use this data, please cite:

 Ryan White, Ashley Eden, Michael Maire, "Automatic Prediction of 
        Human Attractiveness", CS 280 class report, December 2003.
        
Contact datasets@ryanmwhite.com if you have any questions.