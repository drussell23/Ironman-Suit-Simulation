using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace IronManSim.Frontend
{
    /// <summary>
    /// Manages all audio for the Iron Man suit experience
    /// </summary>
    public class AudioManager : MonoBehaviour
    {
        [Header("Audio Sources")]
        [SerializeField] private AudioSource musicSource;
        [SerializeField] private AudioSource sfxSource;
        [SerializeField] private AudioSource voiceSource;
        [SerializeField] private AudioSource ambientSource;
        [SerializeField] private AudioSource engineSource;
        
        [Header("Master Volume")]
        [SerializeField] private float masterVolume = 1f;
        [SerializeField] private float musicVolume = 0.7f;
        [SerializeField] private float sfxVolume = 1f;
        [SerializeField] private float voiceVolume = 1f;
        
        [Header("Boot Sequence")]
        [SerializeField] private AudioClip bootSequenceClip;
        [SerializeField] private AudioClip arcReactorStartup;
        [SerializeField] private AudioClip systemsOnline;
        
        [Header("HUD Sounds")]
        [SerializeField] private AudioClip hudActivate;
        [SerializeField] private AudioClip hudDeactivate;
        [SerializeField] private AudioClip alertSound;
        [SerializeField] private AudioClip targetLockSound;
        [SerializeField] private AudioClip modeChangeSound;
        
        [Header("Weapon Sounds")]
        [SerializeField] private AudioClip repulsorCharge;
        [SerializeField] private AudioClip repulsorFire;
        [SerializeField] private AudioClip missileLaunch;
        [SerializeField] private AudioClip weaponEmpty;
        
        [Header("Flight Sounds")]
        [SerializeField] private AudioClip thrusterLoop;
        [SerializeField] private AudioClip sonicBoom;
        [SerializeField] private AudioClip windLoop;
        
        [Header("Combat Music")]
        [SerializeField] private List<AudioClip> combatTracks;
        [SerializeField] private List<AudioClip> missionTracks;
        [SerializeField] private AudioClip emergencyTrack;
        
        // Audio pools for performance
        private Dictionary<string, Queue<AudioSource>> audioPoolMap;
        private List<AudioSource> activeAudioSources;
        
        // State
        private Coroutine currentMusicFade;
        private bool isEngineActive = false;
        
        #region Initialization
        
        void Awake()
        {
            SetupAudioSources();
            InitializeAudioPools();
        }
        
        private void SetupAudioSources()
        {
            // Create audio sources if not assigned
            if (musicSource == null)
            {
                musicSource = CreateAudioSource("Music", true, false);
                musicSource.volume = musicVolume;
            }
            
            if (sfxSource == null)
            {
                sfxSource = CreateAudioSource("SFX", false, false);
                sfxSource.volume = sfxVolume;
            }
            
            if (voiceSource == null)
            {
                voiceSource = CreateAudioSource("Voice", false, false);
                voiceSource.volume = voiceVolume;
                voiceSource.priority = 0; // Highest priority
            }
            
            if (ambientSource == null)
            {
                ambientSource = CreateAudioSource("Ambient", true, true);
                ambientSource.volume = 0.3f;
            }
            
            if (engineSource == null)
            {
                engineSource = CreateAudioSource("Engine", true, true);
                engineSource.volume = 0.5f;
            }
        }
        
        private AudioSource CreateAudioSource(string name, bool loop, bool playOnAwake)
        {
            GameObject obj = new GameObject($"AudioSource_{name}");
            obj.transform.SetParent(transform);
            
            AudioSource source = obj.AddComponent<AudioSource>();
            source.loop = loop;
            source.playOnAwake = playOnAwake;
            source.spatialBlend = 0f; // 2D sound
            
            return source;
        }
        
        private void InitializeAudioPools()
        {
            audioPoolMap = new Dictionary<string, Queue<AudioSource>>();
            activeAudioSources = new List<AudioSource>();
            
            // Create pools for different sound types
            CreateAudioPool("OneShot", 10);
            CreateAudioPool("Spatial", 5);
        }
        
        private void CreateAudioPool(string poolName, int size)
        {
            Queue<AudioSource> pool = new Queue<AudioSource>();
            
            for (int i = 0; i < size; i++)
            {
                GameObject obj = new GameObject($"PooledAudio_{poolName}_{i}");
                obj.transform.SetParent(transform);
                
                AudioSource source = obj.AddComponent<AudioSource>();
                source.playOnAwake = false;
                obj.SetActive(false);
                
                pool.Enqueue(source);
            }
            
            audioPoolMap[poolName] = pool;
        }
        
        #endregion
        
        #region Boot Sequence
        
        public void PlayBootSequence()
        {
            StartCoroutine(BootSequenceAudio());
        }
        
        private IEnumerator BootSequenceAudio()
        {
            // Arc reactor startup
            if (arcReactorStartup != null)
            {
                sfxSource.PlayOneShot(arcReactorStartup);
                yield return new WaitForSeconds(1f);
            }
            
            // Main boot sequence
            if (bootSequenceClip != null)
            {
                musicSource.clip = bootSequenceClip;
                musicSource.Play();
                
                yield return new WaitForSeconds(bootSequenceClip.length - 1f);
            }
            
            // Systems online
            if (systemsOnline != null)
            {
                sfxSource.PlayOneShot(systemsOnline);
            }
            
            // Start ambient engine
            StartEngineLoop();
        }
        
        #endregion
        
        #region HUD Audio
        
        public void PlayHUDActivate()
        {
            PlaySound(hudActivate, sfxSource);
        }
        
        public void PlayHUDDeactivate()
        {
            PlaySound(hudDeactivate, sfxSource);
        }
        
        public void PlayAlert(AlertLevel level)
        {
            if (alertSound != null)
            {
                AudioSource source = GetPooledAudioSource();
                source.pitch = level == AlertLevel.Critical ? 1.2f : 1f;
                source.volume = sfxVolume;
                source.PlayOneShot(alertSound);
                
                StartCoroutine(ReturnToPool(source, alertSound.length));
            }
        }
        
        public void PlayModeChange()
        {
            PlaySound(modeChangeSound, sfxSource);
        }
        
        public void PlayTargetLock()
        {
            PlaySound(targetLockSound, sfxSource);
        }
        
        #endregion
        
        #region Weapon Audio
        
        public void PlayRepulsorCharge()
        {
            if (repulsorCharge != null)
            {
                AudioSource source = GetPooledAudioSource();
                source.clip = repulsorCharge;
                source.Play();
                
                StartCoroutine(ReturnToPool(source, repulsorCharge.length));
            }
        }
        
        public void PlayRepulsorFire()
        {
            PlaySound(repulsorFire);
        }
        
        public void PlayMissileLaunch()
        {
            PlaySound(missileLaunch);
        }
        
        public void PlayWeaponEmpty()
        {
            PlaySound(weaponEmpty, sfxSource, 0.8f);
        }
        
        #endregion
        
        #region Flight Audio
        
        public void StartEngineLoop()
        {
            if (thrusterLoop != null && !isEngineActive)
            {
                engineSource.clip = thrusterLoop;
                engineSource.Play();
                engineSource.volume = 0f;
                
                StartCoroutine(FadeAudioSource(engineSource, 0.5f, 2f));
                isEngineActive = true;
            }
        }
        
        public void StopEngineLoop()
        {
            if (isEngineActive)
            {
                StartCoroutine(FadeAudioSource(engineSource, 0f, 1f));
                isEngineActive = false;
            }
        }
        
        public void UpdateEngineIntensity(float velocity)
        {
            if (engineSource != null && isEngineActive)
            {
                // Adjust pitch and volume based on velocity
                float normalizedVelocity = Mathf.Clamp01(velocity / 100f);
                engineSource.pitch = 0.8f + normalizedVelocity * 0.4f;
                engineSource.volume = 0.3f + normalizedVelocity * 0.4f;
            }
        }
        
        public void PlaySonicBoom()
        {
            PlaySound(sonicBoom, null, 1.5f);
        }
        
        #endregion
        
        #region Music System
        
        public void PlayCombatMusic()
        {
            if (combatTracks != null && combatTracks.Count > 0)
            {
                AudioClip track = combatTracks[Random.Range(0, combatTracks.Count)];
                CrossfadeMusic(track, 2f);
            }
        }
        
        public void PlayMissionMusic()
        {
            if (missionTracks != null && missionTracks.Count > 0)
            {
                AudioClip track = missionTracks[Random.Range(0, missionTracks.Count)];
                CrossfadeMusic(track, 2f);
            }
        }
        
        public void PlayEmergencyMusic()
        {
            if (emergencyTrack != null)
            {
                CrossfadeMusic(emergencyTrack, 0.5f);
            }
        }
        
        public void StopMusic(float fadeTime = 2f)
        {
            if (currentMusicFade != null)
            {
                StopCoroutine(currentMusicFade);
            }
            
            currentMusicFade = StartCoroutine(FadeAudioSource(musicSource, 0f, fadeTime));
        }
        
        private void CrossfadeMusic(AudioClip newTrack, float fadeTime)
        {
            if (currentMusicFade != null)
            {
                StopCoroutine(currentMusicFade);
            }
            
            currentMusicFade = StartCoroutine(CrossfadeMusicCoroutine(newTrack, fadeTime));
        }
        
        private IEnumerator CrossfadeMusicCoroutine(AudioClip newTrack, float fadeTime)
        {
            // Fade out current
            yield return StartCoroutine(FadeAudioSource(musicSource, 0f, fadeTime / 2f));
            
            // Switch track
            musicSource.clip = newTrack;
            musicSource.Play();
            
            // Fade in new
            yield return StartCoroutine(FadeAudioSource(musicSource, musicVolume, fadeTime / 2f));
        }
        
        #endregion
        
        #region Audio Utilities
        
        private void PlaySound(AudioClip clip, AudioSource source = null, float volumeScale = 1f)
        {
            if (clip == null) return;
            
            if (source != null)
            {
                source.PlayOneShot(clip, volumeScale);
            }
            else
            {
                // Use pooled source
                AudioSource pooledSource = GetPooledAudioSource();
                pooledSource.volume = sfxVolume * volumeScale;
                pooledSource.PlayOneShot(clip);
                
                StartCoroutine(ReturnToPool(pooledSource, clip.length));
            }
        }
        
        public void PlaySpatialSound(AudioClip clip, Vector3 position, float volumeScale = 1f)
        {
            if (clip == null) return;
            
            AudioSource source = GetPooledAudioSource("Spatial");
            source.transform.position = position;
            source.spatialBlend = 1f; // 3D sound
            source.volume = sfxVolume * volumeScale;
            source.PlayOneShot(clip);
            
            StartCoroutine(ReturnToPool(source, clip.length));
        }
        
        private AudioSource GetPooledAudioSource(string poolName = "OneShot")
        {
            if (audioPoolMap.ContainsKey(poolName) && audioPoolMap[poolName].Count > 0)
            {
                AudioSource source = audioPoolMap[poolName].Dequeue();
                source.gameObject.SetActive(true);
                activeAudioSources.Add(source);
                return source;
            }
            
            // Create new if pool is empty
            return CreateAudioSource($"DynamicPooled_{poolName}", false, false);
        }
        
        private IEnumerator ReturnToPool(AudioSource source, float delay)
        {
            yield return new WaitForSeconds(delay);
            
            source.Stop();
            source.clip = null;
            source.gameObject.SetActive(false);
            
            activeAudioSources.Remove(source);
            
            // Determine pool
            string poolName = source.spatialBlend > 0 ? "Spatial" : "OneShot";
            if (audioPoolMap.ContainsKey(poolName))
            {
                audioPoolMap[poolName].Enqueue(source);
            }
        }
        
        private IEnumerator FadeAudioSource(AudioSource source, float targetVolume, float duration)
        {
            float startVolume = source.volume;
            float elapsed = 0f;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                source.volume = Mathf.Lerp(startVolume, targetVolume, elapsed / duration);
                yield return null;
            }
            
            source.volume = targetVolume;
            
            if (targetVolume == 0f && source.isPlaying)
            {
                source.Stop();
            }
        }
        
        #endregion
        
        #region Volume Control
        
        public void SetMasterVolume(float volume)
        {
            masterVolume = Mathf.Clamp01(volume);
            AudioListener.volume = masterVolume;
        }
        
        public void SetMusicVolume(float volume)
        {
            musicVolume = Mathf.Clamp01(volume);
            if (musicSource != null)
            {
                musicSource.volume = musicVolume;
            }
        }
        
        public void SetSFXVolume(float volume)
        {
            sfxVolume = Mathf.Clamp01(volume);
            if (sfxSource != null)
            {
                sfxSource.volume = sfxVolume;
            }
        }
        
        public void SetVoiceVolume(float volume)
        {
            voiceVolume = Mathf.Clamp01(volume);
            if (voiceSource != null)
            {
                voiceSource.volume = voiceVolume;
            }
        }
        
        #endregion
        
        #region Cleanup
        
        void OnDestroy()
        {
            // Stop all coroutines
            StopAllCoroutines();
            
            // Stop all audio
            foreach (var source in activeAudioSources)
            {
                if (source != null)
                {
                    source.Stop();
                }
            }
        }
        
        #endregion
    }
}