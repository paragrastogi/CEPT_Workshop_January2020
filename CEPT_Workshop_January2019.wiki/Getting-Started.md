<h1>Getting started</h1>

I will assume you have a Windows computer and an internet connection. You can use most of these commands in a Linux command line as well - just make sure you get Linux distros from the web instead of Windows ones! I have developed these scripts using Python 3.6.3 and Jupyter 4.3.0.

<h2>1. Installing Python</h2>

Do you already have Python 3 (any version, but preferably 3.6) installed? Great, you're winning already. Skip ahead to the next section.

If not, go to the [conda website](https://conda.io/miniconda.html) and install the 64-bit version for Python 3.6. Anaconda is just one distribution of Python (and R) - feel free to install the latest basic python from the project's [own website](https://www.python.org/downloads/windows/).

**NB:** I will use the command `conda install` to install packages below. If you would like to use the basic python distro, then you must use `pip install`. 

Open the command prompt by pressing `Windows`+`R` and typing `cmd` in the little window that appears. The command prompt is that dank black screen with a blinking cursor that all the strange computer people seem to have on their screens, shown [here](https://en.wikipedia.org/wiki/Cmd.exe) in all its glory. Your parents or professors might call it the "MS-DOS prompt". Ignore them, they are wrong.

In the command prompt, type: 

<kbd>python --version</kbd>

You should see something like this:

<kbd>Python 3.6.3 :: Anaconda, Inc.</kbd>

If you don't see this or you see an error, it could mean one of two things: you haven't installed Python or the command `python` is not in your `PATH` variable. See this <a href='https://superuser.com/questions/143119/how-to-add-python-to-the-windows-path'>helpful link</a> to resolve the latter issue.

Set up with Python? Good! Proceed to step 2.


<h2>2. Installing Python modules</h2>
<h4>AKA Command Line Superpowers</h4>

We will use the command prompt to install the modules/packages/libraries needed for this tutorial. Use the following commands, in no particular order: <ul>
<li> <kbd>conda install jupyter</kbd>
<li> <kbd>conda install numpy</kbd>
<li> <kbd>conda install scipy</kbd>
<li> <kbd>conda install pandas</kbd>
<li> <kbd>conda install sklearn</kbd>
<li> <kbd>conda install matplotlib</kbd>
<li> <kbd>conda install seaborn</kbd>
<li> <kbd>conda install h5py</kbd>
</ul>

If all of these modules installed fine, you are all set to start with the notebooks for this workshop. You can check your installation by using the `conda install <module name>` command again. If the module is already installed, you will get a message saying `All requested packages already installed.`.

Now, how do you actually download all these scripts and notebooks to your computer? 

<h2>3. Downloading this repository</h2>

In your browser, go to the <a href='https://github.com/paragrastogi/CEPT_Weather_Uncertainty'>github repo</a>. On that page, you should see a green button labelled `Clone or download`. Click on that to see options. The easiest is to download the repository as a ZIP file. You can also copy the git link (same as the web address but ending in `.git`) and use git on your computer to `clone` or `fork` the repo. See this <a href='https://git-scm.com/download/win'>webpage</a> for instructions on how to download and use git. Don't be afraid to use the command line interface for big girls/boys.

<h2>4. Running the IPython notebook</h2>

In the command prompt, navigate to the folder where you have cloned/downloaded this repository. Navigate to the `scripts` directory. Run a `dir` command to see if the IPython notebook exists in the scripts directory. If it does exist, and it should, type the following command: 

<kbd>jupyter notebook tutorial.ipynb</kbd>

You will see a bunch of time-stamped commands start flowing in the command prompt. Your default browser (I use Firefox) will fire up and the notebook will open in a new tab. **Do not close the command prompt where you started the notebook while you are working on it.** This will shut down the python server for your notebook and it will no longer work.