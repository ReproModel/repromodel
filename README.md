<p align = "center">
  <img alt = "ReproModel - Open Source Toolbox for Boosting the AI Research Efficiency" src = "./public/readme-files/quick-overview.gif" width = "75%">
</p>

<h3 align = "center">ReproModel</h3>

<p align = "center">Open Source Toolbox for Boosting the AI Research Efficiency</p>

<div align = "center">
  <a href = "https://github.com/ReproModel/repromodel/stargazers"><img alt = "GitHub Repo Stars" src = "https://img.shields.io/github/stars/ReproModel/repromodel"></a>
  <a href = "https://github.com/ReproModel/repromodel/blob/dev/LICENSE.md"><img alt = "License" src = "https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href = "https://discord.gg/CdJzm8zmpu"><img alt = "Discord" src = "https://img.shields.io/badge/Discord-3-brightgreen?logo=discord&logoColor=white"></a>
</div>

<br/>

**ReproModel**  helps the AI research community to <strong>reproduce, compare, train, and test AI models</strong> faster.

ReproModel toolbox revolutionizes research efficiency by providing standardized models, dataloaders, and processing procedures. It features a comprehensive suite of pre-existing experiments, a code extractor, and an LLM descriptor. This toolbox <strong>allows researchers to focus on new datasets and model development</strong>, significantly reducing time and computational costs.

With this <strong>no-code solution</strong>, you'll have access to a collection of benchmark and SOTA models and datasets. Dive into training visualizations, effortlessly extract code for publication, and let our LLM-powered automated methodology description writer do the heavy lifting.

The current prototype helps researchers to modularize their development and compare the performance of <strong>each step in the pipeline in a reproducible way</strong>. This prototype version helped us reduce the time for model development, computation, and writing by at least 40%. Watch our [demo](https://www.youtube.com/watch?v=MQHZMEloUps).

The coming versions will help researchers build upon state-of-the-art research faster by just <strong>loading the previously published study ID</strong>. All code, experiments, and results will already be verified and stored in our system.

https://repromodel.netlify.app

## Features
:white_check_mark: Standard Models Included<br/>
:white_check_mark: Known Datasets<br/>
:white_check_mark: Metrics (100+)<br/>
:white_check_mark: Losses (20+)<br/>
:white_check_mark: Data Splitting<br/>
:white_check_mark: Augmentations<br/>
:white_check_mark: Optimizers (10+)<br/>
:white_check_mark: Learning Rate Schedulers<br/>
:white_check_mark: Early Stopping Criterion<br/>
:white_check_mark: Training Device Selection<br/>
:white_check_mark: Logging (Tensorboard ...)<br/>
:white_check_mark: AI Experiment Description Generator<br/>
:white_check_mark: Code Extractor<br/>
:white_check_mark: Custom Script Editor<br/>
:white_check_mark: Docker image<br/>
:black_square_button: GUI augmentation builder<br/>
:black_square_button: Conventional ML models workflow<br/>
:black_square_button: Parallel training<br/>
:black_square_button: Statistical testing<br/>

## Documentation
For examples and step-by-step instructions, please visit our full documentation at https://www.repromodel.com/docs.

## Running Locally
### Docker (Option 1)
Please verify that you have [Docker](https://www.docker.com/get-started/) or Docker CLI installed on your system.<br>

Pull the docker image:<br>
`docker pull dsitnik1612/repromodel`<br>

Run the container:<br>
`docker run --name ReproModel -p 5173:5173 -p 6006:6006 -p 5005:5005 repromodel`<br>

Then open the frontend under: http://localhost:5173/

### Source code (Option 2)
In case you want to run the ReproModel directly from the source code, here are the steps:<br>

You will need to have [Node.js](https://nodejs.org) installed.<br/> 

**Combines npm install, creation of a virtual environment, as well as the launch of the frontend and backend:**
<br>
```
npm run repromodel            // Mac and Linux
npm run repromodel-windows    // Windows
```

**If you want to launch the frontend only:**
```
npm install
npm run dev
```

**For using the Methodology Generator, you need to have Ollama installed**
You can get Ollama from their [website](https://ollama.com/download) and pull the model of your choice.
```
npm install
npm run repromodel-with-llm            // Mac and Linux
npm run repromodel-with-llm-windows    // Windows
```

Then open the frontend under: http://localhost:5173/

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create.

Any contributions you make are greatly appreciated. If you have a suggestion that would make this better, please read our [Contribution Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

### List of contributors
<table>
  <tr>
    <td align = "center">
      <img src = "https://avatars.githubusercontent.com/u/13439539" width = "100px" alt = "Dario Sitnik"/><br/>
      <sub><b>Dario Sitnik, PhD</b></sub><br/>
      <sub>AI Scientist</sub><br/>
      <a href="https://github.com/dsitnik">GitHub</a>
    </td>
    <td align = "center">
      <img src = "https://avatars.githubusercontent.com/u/168817578" width = "100px" alt = "Mint Owl"/><br/>
      <sub><b>Mint Owl</b></sub><br/>
      <sub>ML Engineer</sub><br/>
      <a href="https://github.com/mintowltech">GitHub</a>
    </td>
    <td align = "center">
      <img src = "https://avatars.githubusercontent.com/u/168830779" width = "100px" alt = "Martin Schumakher"/><br/>
      <sub><b>Martin Schumakher</b></sub><br/>
      <sub>Developer</sub><br/>
      <a href = "https://github.com/martinschum">GitHub</a>
    </td>
    <td align = "center">
      <img src = "https://avatars.githubusercontent.com/u/20110627" width = "100px" alt = "Tomonari Feehan"/><br/>
      <sub><b>Tomonari Feehan</b></sub><br/>
      <sub>Developer</sub><br/>
      <a href = "https://github.com/tomonarifeehan">GitHub</a>
    </td>
  </tr>
</table>

### Contributor of the month - June 2024
<img src = "https://avatars.githubusercontent.com/u/20110627" width = "100px" alt = "Tomonari Feehan"/><br/>
<sub><b>Tomonari Feehan</b></sub><br/>
<sub>Developer</sub><br/>
<a href = "https://github.com/tomonarifeehan">GitHub</a>

## Stats
![Alt](https://repobeats.axiom.co/api/embed/4595dc84d9863dd36d50711c399685b444e54d0e.svg "Repobeats Analytics Image")

## Questions & Support
For questions or any type of support, you can reach out to me via dario.sitnik@gmail.com

## License
This project is licensed under the [MIT License](LICENSE.md).
<br/>

![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
