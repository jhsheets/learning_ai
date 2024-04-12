# Before running
You need to create a volume called **ai-learning-llm-cache**

You can do this by running the following command: 
```
docker volume create ai-learning-llm-cache
```

These examples need LLMs to run. To make things simpler, the `docker-compose.yml` files will download the LLM file it 
needs when it runs, if the file doesn't exist.

These are very large files. This volume will allow us to share the LLMs between examples, saving your bandwidth, and
greatly reducing the first time it will take apps to start (if the LLM was previously downloaded).

> You have to manually delete this volume.
> You can run the following command to do this: 
> ```
> docker volume remove ai-learning-llm-cache
> ```

# Running examples
Most of these examples use `gradio` as a web frontend.

In most cases, the webapp won't start until the app has the LLM it needs to run.
As stated above, we share this directory across the different `docker-compose.yml` files in the 
**ai-learning-llm-cache** to help speed this up