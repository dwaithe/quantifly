//Macro written by Dominic Waithe to extract circular regions from Drosopholia egg grid.
//Open a grid image and then click the 'Run' button to activate the macro.
//Once the macro is correct. Save the resulting stack as a 'tif' file on your computer
//The resulting '.tif' file can then be analysed using QuantiFly.
num_im_per_row = 4;

//Run the script in headless mode to speed up processing.
setBatchMode(true);

//Sets the Imagej preferences so that they are consistent between versions.
run("Colors...", "foreground=white background=black selection=yellow");
run("Set Measurements...", "area min centroid center bounding redirect=None decimal=3");

//Creates a duplicate of the original which we then resize.
run("Duplicate...", "title=downscale");
run("Size...","width="+round(getWidth()/2)+" height="+round(getHeight()/2)+" constrain average interploation=Bilinear");
//Makes a duplicate which we threshold
run("Duplicate...", "title=thr");
//Necessary for thresholding
run("Invert");
run("8-bit");

//We use an automatic method known as Otsu to threshold.
setAutoThreshold("Otsu dark");
run("Convert to Mask");

//Clear the ROImanager to remove any regions.
roiManager("Reset");
//We then find the iamge regions.
run("Analyze Particles...", "size=0.1-Infinity circularity=0.40-1.00 display add");
roiCount = roiManager('Count');
selectWindow("downscale");

//Next we run through each region, calculate the average size and then calculate
//the best order so it runs from top-left to bottom right, one row at a time.

final_rank = newArray(roiCount);
wid= 0;
hei= 0;

for (rc=0;rc<roiCount; rc =rc+4){
	selectWindow("downscale");
	left = 10000;
	grX = newArray(4);
	
	for (id1=0;id1<4;id1++){
		
	roiManager("select",rc+id1);
	run('Measure');	
	grX[id1] = getResult('X',nResults-1);
	wid += getResult("Width",nResults-1);
	hei += getResult("Height",nResults-1);
	
	
	}
	ranks = Array.rankPositions(grX);
	for (rk=0;rk<4;rk++){
	final_rank[rc+rk]  = rc +ranks[rk];
	}
	}

//Calculate the average size of an image.
ave_wid = wid/roiCount;
ave_hei = hei/roiCount;

//Add the image regions in order.
for (rc=0;rc<roiCount; rc ++){
	selectWindow("downscale");
	roiManager("select",final_rank[rc]);
	
	run('Measure');
	grX = getResult('X',nResults-1);
	grY = getResult('Y',nResults-1);
	run("Specify...", "width="+ave_wid+" height"+ave_hei+" x="+grX+" y="+grY+" oval centered scaled");
	roiManager("Add");
	
	}
//Clear the outside region to simplify subsequent analysis
selectWindow("downscale");
roiManager('Select All');
roiManager('OR');
run('Clear Outside');
//We then extract the regions in the correct order.
for (rc=roiCount;rc<roiCount*2; rc ++){
	selectWindow("downscale");
	roiManager("select",rc);
	run("Duplicate...", "title=region_"+rc);
	}


//We then combine the regions into a stack and give it the name regions.
//This is the image stack which we want to save.
run("Images to Stack", "method=[Copy (center)] name=Regions title=region use");
setBatchMode(false);