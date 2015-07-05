<html>


<head>
</head>
<body>
<H1>QuantiFly</H1>
<p> QuantiFly is a software package for automating egg counting in  <i>Drosophila</i> genetics research[1]. QuantiFly is optimised for counting eggs in media containing vials, captured using a digital USB camera. QuantiFly utilises an advanced computer vison and machine learning algorithm[2][3] and improves upon the baseline approach [1]. The technique uses a modified version of a density estimation approach which is a robust and accurate means of estimating numbers of objects in dense and complicated scenes.
<H2>Source code</H2>
<p>QuantiFly is open source and the python code files are included above. QuantiFly has multiple python dependencies.</p>

<H2>QuantiFly Software</H2>
<p> The software which is available through the following link allows immediate access to the QuantiFly technique:
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/download_page.html#FoCuS">Click for downloads</a></p>
<H3>Manual</H3>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/software/howTo.pdf">pdf manual for using QuantiFly</p>
<H3>Datasets</H3>
<p> The following datasets provide examples of the images on which QuantiFly was designed to operate </p>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/ground_truth_data/data01-20130531-DM.zip">Dataset A</a> transparent media. </p>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/ground_truth_data/data02-20130709-DM.zip">Dataset B</a> transparent media. </p>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/ground_truth_data/data03-20140331-DM.zip">Dataset C</a> transparent media. </p>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/ground_truth_data/data04-20140331-DM.zip">Dataset D</a> transparent media. </p>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/ground_truth_data/data05-bias-DM.zip">Dataset E</a> transparent media. </p>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/ground_truth_data/data06-20130704-SY.zip">Dataset F</a> opaque media. </p>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/ground_truth_data/data07-20130709-SY.zip">Dataset G</a> opaque media. </p>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/ground_truth_data/data08-20140409-SY.zip">Dataset H</a> opaque media. </p>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/ground_truth_data/data09-20140409-SY.zip">Dataset I</a> opaque media. </p>
<p><a href ="http://sara.molbiol.ox.ac.uk/dwaithe/ground_truth_data/data10-bias-SY.zip">Dataset J</a> opaque media. </p>

<H3>FAQ</H3>
<p>Q: I've double-clicked the software and it takes a while to load? A: The first time the software is run, it may take a little while to appear, the next time it will load almost instantly.</p>

<p>Q: What image types can I use with QuantiFly? A: QuantiFly is currently compatible with '.png' and '.tif' files. QuantiFly will also work '.tif' stacks but only if the colour space is RGB. If in doubt, download Fiji (<a href="fiji.sc/Downloads">fiji.sc/Downloads</a>) and convert your image to RGB and File->SaveAs 'tif'.

<p>Q: I notice the datasets contain image files of two types, what are the ones with dot in the name? A: These are the ground-truth images which have been produced by a human researcher to analyse the accuracy of the technique. Each image contains a variety of dots which are spatially located to represent the egg in the corresponding image. To find the overall count you just need to count the dots. This can be done using Fiji in three steps: Open the image. Click from the menu Process->Find Maxima. Check 'Preview point selection' to see the overall count.

<p>Q: Can QuantiFly be used to count other things?  A: Yes QuantiFly can be used to count just about anything of more-or-less constant size in 2D images.</p>

<p> Dominic Waithe 2015 (c)</p>
<p>[1] Waithe, Dominic, et al. "QuantiFly: Robust Trainable Software for Automated Drosophila Egg Counting." (2015): e0127659.</p>
<p>[2] Lempitsky, Victor, and Andrew Zisserman. "Learning to count objects in images." Advances in Neural Information Processing Systems. 2010.</p>
<p>[3] Fiaschi, Luca, et al. "Learning to count with regression forest and structured labels." Pattern Recognition (ICPR), 2012 21st International Conference on. IEEE, 2012.</p>

</body>
