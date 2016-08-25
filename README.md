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
<p><a href ="https://github.com/dwaithe/quantifly/releases/tag/2.0">Click for downloads</a></p>
<H3>Manual</H3>
<p> A detailed manual for QuantiFly can be downloaded here:</p>
<p><a href ="https://github.com/dwaithe/quantifly/releases/tag/2.0">Click for downloads</a></p>
<H3>Datasets</H3>
<p> QuantiFly has eight test datasets which can be downloaded from the following page:</p>
<p><a href ="https://github.com/dwaithe/quantifly/releases/tag/2.0">Click for downloads</a></p>

<H3>FAQ</H3>
<p>Q: I have found a bug in the code how can I report it? A: Goto <a href="https://github.com/dwaithe/quantifly/issues">https://github.com/dwaithe/quantifly/issues</a>. You can report issues or ask any questions here.</p>
<p>Q: I've double-clicked the software and it takes a while to load? A: The first time the software is run, it may take a little while to appear, the next time it will load almost instantly.</p>

<p>Q: What image types can I use with QuantiFly? A: QuantiFly is currently compatible with '.png' and '.tif' files. QuantiFly will also work '.tif' stacks but only if the colour space is RGB. If in doubt, download Fiji (<a href="fiji.sc/Downloads">fiji.sc/Downloads</a>) and convert your image to RGB and File->SaveAs 'tif'.

<p>Q: I notice the datasets contain image files of two types, what are the ones with dot in the name? A: These are the ground-truth images which have been produced by a human researcher to analyse the accuracy of the technique. Each image contains a variety of dots which are spatially located to represent the egg in the corresponding image. To find the overall count you just need to count the dots. This can be done using Fiji in three steps: Open the image. Click from the menu Process->Find Maxima. Check 'Preview point selection' to see the overall count.

<p>Q: Can QuantiFly be used to count other things?  A: Yes QuantiFly can be used to count just about anything of more-or-less constant size in 2D images.</p>

<p> Dominic Waithe 2015 (c)</p>
<p>[1] Waithe, Dominic, et al. "QuantiFly: Robust Trainable Software for Automated Drosophila Egg Counting." (2015): e0127659.</p>
<p>[2] Lempitsky, Victor, and Andrew Zisserman. "Learning to count objects in images." Advances in Neural Information Processing Systems. 2010.</p>
<p>[3] Fiaschi, Luca, et al. "Learning to count with regression forest and structured labels." Pattern Recognition (ICPR), 2012 21st International Conference on. IEEE, 2012.</p>

</body>
