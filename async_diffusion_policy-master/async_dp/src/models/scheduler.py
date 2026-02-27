from diffusers import DDPMScheduler, DDIMScheduler

def get_scheduler(name='ddim', num_train_timesteps=100, num_inference_steps=16):
    beta_schedule = 'squaredcos_cap_v2'
    clip_sample = True
    
    if name == 'ddpm':
        return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule, clip_sample=clip_sample)
    elif name == 'ddim':
        scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule, clip_sample=clip_sample)
        scheduler.set_timesteps(num_inference_steps)
        return scheduler
    else: raise ValueError(f"Unknown scheduler: {name}")