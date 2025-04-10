# CO4: Imaging for the Life Sciences

![Imaging for the Life Sciences](data/imaging4ls-logo-1.jpg)



Git repository for the MSLS module CO4 – Imaging for the Life Sciences ([Moodle](https://mslscommunitycentre.ch/course/view.php?id=133))



## Instructions – Overview

In the practical part, we will use Python and Jupyter notebooks to work hands-on with image data. We recommend the following setup:

* Python 3.10 or higher
* Jupyter 7.x
* Microsoft Visual Studio Code (with Python and Jupyter extensions)
* Git version control
* A set of Python packages (see requirements.txt)

We also recommend using a tool to manage different Python environments, such as [venv](https://docs.python.org/3/library/venv.html) (or [Anaconda](https://www.anaconda.com/)). VS Code supports both methods.



## Instructions – Step by step:

* Install **Python**

  * There are different possibilities to install Python on your system
  * One possibility is to get the installer from: https://www.python.org/

* Install **Visual Studio Code** (VS Code)

  * Get the installer from https://code.visualstudio.com/

* Install the official **Python extension** for VS Code

  * Open VS Code
  * Click on the extensions view icon on the sidebar  
    (shortcut: `Ctrl/Cmd+Shift+X`)
  * Search the extension "*Python*"
  * Click *Install* on the Python extension by Microsoft
  * If you are not familiar with the Python interface: Read [this tutorial](https://code.visualstudio.com/docs/languages/python)

* Install the official **Jupyter extension** for VS Code

  * Search the extension "*Jupyter*"
  * Click *Install* on the Jupyter extension by Microsoft
  * If you are not familiar with the Jupyter interface: Read [this tutorial](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

* Install the version control system **git**:

  * Follow [these instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

* Clone the **course repository**: Open the following lines in a terminal

  ```bash
  # Navigate to the folder where you want to store 
  # the course materials for the CO4 module.
  #   - Use mkdir to create the folder first
  #   - Use "" if your path contains spaces!
  
  # Win:
  cd "path\to\folder"
  # Mac / Linux:
  cd "path/to/folder"
  
  # Clone this repository
  git clone https://github.com/hirsch-lab/msls-co4-ss25.git
  
  ```

* Create a virtual Python environment (venv):

  * Make sure that you are running the desired Python:
    ```bash
    python --version
    ```

  * Create a venv named `venv-msls-co4`:
    ```bash
    python -m venv "venv-msls-co4"
    ```

  * Activate the new environment
    ```bash
    # Win:
    venv-msls-co4\scripts\activate
    
    # Mac / Linux:
    source venv-msls-co4/bin/activate
    ```

  * Update the Python package installer
    ```bash
    python -m pip install pip --upgrade
    ```

  * Install the required packages
    ```bash
    # Win
    python -m pip install -r msls-co4-ss25\requirements.txt
    
    # Mac / Linux
    python -m pip install -r msls-co4-ss25/requirements.txt
    ```

* Add the Git project folder to your VS Code **workspace**:

  * Open VS Code
  * File $\rightarrow$ Add Folder to Workspace...
  * Choose the folder `msls-co4-ss25`

* Run the **test notebook** in VS Code:

  * Open VS Code 
  * Open the notebook: `notebooks/00-test-setup.ipynb`
  * When running the Jupyter notebook for the first time, you need to select the correct  **Jupyter kernel**:
    * Open the command palette: Ctrl/Cmd + Shift + P
    * Type and select: *Notebook: Select Notebook Kernel*
    * Choose: *Select Another Kernel*
    * Choose: *Python Environments*
    * Choose `venv-msls-co4` from the drop down list  
      If you don't see the new venv, you may have to restart VS Code
  * Run the notebook 

<!--Unfortunately, relative links inside lists (and other non-paragraph markdown blocks) are currently not supported. See here: https://github.com/orgs/community/discussions/67750, https://github.com/github/markup/issues/1773 -->
