# ViennaUp AI for Sustainability Hackathon '23

## Challenge [Automated Recycling](https://sustainista.net/aiforsustainability/)
- Input
    - images from a custom hardware scanner
    - some data samples are already provided (via AWS)
    - additional data can be collected with the available hardware
    - *seems only to be camera images (no fancy modalities like uv-scanner etc.)*
- AI model
    - self-learning solution that balances performance vs. time to deploy.
- Business aspects
    - What business model patterns might be applied? Please develop a promising business model. 
    - *Not clear if it is aimed to be a B2B or B2C solution*
- User Experience
    - Please design your solution and interface.
    - *image driven UI, in python or javascript?*
    - *UI for end-user (B2C) or expert user (B2B)?*
### Comments on the specification

- *I think they want us to design a system that can be deployed to customers and customers can easily perform a fine-tuning step*
    - *e.g.: deployed to MA48 and they can provide sample images and the system quickly learns how to classify certain trash objects and gives rich feedback how to system performs*
- *the two key components seem to be the **self-learning capability** as well as **communicating the models performance to the user** (visualizing metrics, fail-cases, problem classes etc.)*

## Milestones during hackathon

- understanding the problem
    - in the first hours we focus on this
    - we need more information to understand the problem and clearify some a assumptions we made (e.g. B2B vs. B2C)
    - **achieved when:** We know what the challenge is and what the want from us
- finding a suitable model & training process 
    - the more we know how the self-learning aspect could look like the sooner we can work on UX, prototype and business plan
    - **achieved when:** We approximately know how our self-learning system could work & how we want the user to interact with it
- user communication & visualization idea
    - we have to think of how we want to visualize 
    - **achieved when:** we know how to visualize validation performance and inference to the user?

- prototype
    - we need a working prototype that shows how to interact with our system
        - e.g. in order to retrain the model and get results how the model performs
    - **achieved when:** we have integrated the model with a working UI that we can live demonstrate during our pitch
- coming up with a suitable business pattern
    - after we know if B2B or B2C and how the solution should approximately look like we can think about business plans
    - **achieved when:** we have a idea how to produce revenue with our solution, provided some case studies as well as calculated possible market size etc.
- pitch presentation
    - need to clearify if live working prototype is required or not
    - **achieved when:** we pitched that thing

## responsibilities among the team

- research model & self-learning process
    - Lisa, Reza, Matthias
- AWS infrastructure & deployment
    - Reza
- metrics & validation
    - Herwig, Reza, Matthias
- visualization of metrics
    - Florian P., Herwig
- user interface (& UX)
    - Florian P., Herwig, Florian K., Tai
- end-to-end integration
    - Matthias, Florian P.
- business plan
    - Tai, Lisa 
- pitch planning
    - Lisa, Reza, Florian K., (Tai)
- pitch presentation 
    - Lisa, Florian K.


## collection of ideas

random thoughts that are worth to be considered in future

- gamification to improve the UX
    - e.g. supermarket deploys the hardware and offers discount for heavy users (who recycle a lot the correct way?)
- Saliency Map (and other XAI techniques for UX & visualization?)
- use LoRA for self-learning/fine-tuning (--> more stable, less catastrophic forgetting)
