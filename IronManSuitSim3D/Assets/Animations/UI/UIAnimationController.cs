using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;

namespace IronManSim.UI.Animations
{
    /// <summary>
    /// Master controller for all UI animations in the Iron Man suit
    /// Handles coordination between different animation systems
    /// </summary>
    public class UIAnimationController : MonoBehaviour
    {
        [Header("Animation Settings")]
        [SerializeField] private float defaultAnimationSpeed = 1f;
        [SerializeField] private AnimationCurve defaultEasingCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
        
        [Header("Global Animation States")]
        [SerializeField] private bool enableHolographicNoise = true;
        [SerializeField] private float holographicNoiseIntensity = 0.02f;
        [SerializeField] private bool enableGlitchEffects = true;
        
        private Dictionary<string, Coroutine> activeAnimations = new Dictionary<string, Coroutine>();
        private static UIAnimationController instance;
        
        public static UIAnimationController Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = FindObjectOfType<UIAnimationController>();
                }
                return instance;
            }
        }
        
        void Awake()
        {
            if (instance == null)
            {
                instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else if (instance != this)
            {
                Destroy(gameObject);
            }
        }
        
        #region Core Animation Methods
        
        /// <summary>
        /// Animate a float value over time
        /// </summary>
        public void AnimateFloat(string animationId, float startValue, float endValue, float duration, 
            System.Action<float> onUpdate, System.Action onComplete = null, AnimationCurve customCurve = null)
        {
            StopAnimation(animationId);
            activeAnimations[animationId] = StartCoroutine(AnimateFloatCoroutine(
                startValue, endValue, duration, onUpdate, onComplete, customCurve ?? defaultEasingCurve));
        }
        
        /// <summary>
        /// Animate a Vector3 value over time
        /// </summary>
        public void AnimateVector3(string animationId, Vector3 startValue, Vector3 endValue, float duration,
            System.Action<Vector3> onUpdate, System.Action onComplete = null, AnimationCurve customCurve = null)
        {
            StopAnimation(animationId);
            activeAnimations[animationId] = StartCoroutine(AnimateVector3Coroutine(
                startValue, endValue, duration, onUpdate, onComplete, customCurve ?? defaultEasingCurve));
        }
        
        /// <summary>
        /// Animate a Color value over time
        /// </summary>
        public void AnimateColor(string animationId, Color startColor, Color endColor, float duration,
            System.Action<Color> onUpdate, System.Action onComplete = null, AnimationCurve customCurve = null)
        {
            StopAnimation(animationId);
            activeAnimations[animationId] = StartCoroutine(AnimateColorCoroutine(
                startColor, endColor, duration, onUpdate, onComplete, customCurve ?? defaultEasingCurve));
        }
        
        /// <summary>
        /// Stop a specific animation
        /// </summary>
        public void StopAnimation(string animationId)
        {
            if (activeAnimations.ContainsKey(animationId) && activeAnimations[animationId] != null)
            {
                StopCoroutine(activeAnimations[animationId]);
                activeAnimations.Remove(animationId);
            }
        }
        
        /// <summary>
        /// Stop all active animations
        /// </summary>
        public void StopAllAnimations()
        {
            foreach (var animation in activeAnimations.Values)
            {
                if (animation != null)
                {
                    StopCoroutine(animation);
                }
            }
            activeAnimations.Clear();
        }
        
        #endregion
        
        #region Animation Coroutines
        
        private IEnumerator AnimateFloatCoroutine(float start, float end, float duration, 
            System.Action<float> onUpdate, System.Action onComplete, AnimationCurve curve)
        {
            float elapsed = 0f;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime * defaultAnimationSpeed;
                float t = Mathf.Clamp01(elapsed / duration);
                float curveValue = curve.Evaluate(t);
                float currentValue = Mathf.Lerp(start, end, curveValue);
                
                onUpdate?.Invoke(currentValue);
                yield return null;
            }
            
            onUpdate?.Invoke(end);
            onComplete?.Invoke();
        }
        
        private IEnumerator AnimateVector3Coroutine(Vector3 start, Vector3 end, float duration,
            System.Action<Vector3> onUpdate, System.Action onComplete, AnimationCurve curve)
        {
            float elapsed = 0f;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime * defaultAnimationSpeed;
                float t = Mathf.Clamp01(elapsed / duration);
                float curveValue = curve.Evaluate(t);
                Vector3 currentValue = Vector3.Lerp(start, end, curveValue);
                
                onUpdate?.Invoke(currentValue);
                yield return null;
            }
            
            onUpdate?.Invoke(end);
            onComplete?.Invoke();
        }
        
        private IEnumerator AnimateColorCoroutine(Color start, Color end, float duration,
            System.Action<Color> onUpdate, System.Action onComplete, AnimationCurve curve)
        {
            float elapsed = 0f;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime * defaultAnimationSpeed;
                float t = Mathf.Clamp01(elapsed / duration);
                float curveValue = curve.Evaluate(t);
                Color currentValue = Color.Lerp(start, end, curveValue);
                
                onUpdate?.Invoke(currentValue);
                yield return null;
            }
            
            onUpdate?.Invoke(end);
            onComplete?.Invoke();
        }
        
        #endregion
        
        #region Preset Animations
        
        /// <summary>
        /// Fade in a UI element
        /// </summary>
        public void FadeIn(CanvasGroup canvasGroup, float duration = 0.3f, System.Action onComplete = null)
        {
            AnimateFloat($"FadeIn_{canvasGroup.GetInstanceID()}", 
                canvasGroup.alpha, 1f, duration, 
                (value) => canvasGroup.alpha = value, 
                onComplete);
        }
        
        /// <summary>
        /// Fade out a UI element
        /// </summary>
        public void FadeOut(CanvasGroup canvasGroup, float duration = 0.3f, System.Action onComplete = null)
        {
            AnimateFloat($"FadeOut_{canvasGroup.GetInstanceID()}", 
                canvasGroup.alpha, 0f, duration, 
                (value) => canvasGroup.alpha = value, 
                onComplete);
        }
        
        /// <summary>
        /// Scale bounce animation
        /// </summary>
        public void ScaleBounce(Transform target, float duration = 0.5f, float bounceScale = 1.2f)
        {
            string animId = $"ScaleBounce_{target.GetInstanceID()}";
            Vector3 originalScale = target.localScale;
            
            AnimateVector3(animId + "_up", originalScale, originalScale * bounceScale, duration * 0.3f,
                (value) => target.localScale = value,
                () =>
                {
                    AnimateVector3(animId + "_down", target.localScale, originalScale, duration * 0.7f,
                        (value) => target.localScale = value,
                        null,
                        AnimationCurve.EaseInOut(0, 0, 1, 1));
                },
                AnimationCurve.EaseInOut(0, 0, 1, 1));
        }
        
        /// <summary>
        /// Slide in from direction
        /// </summary>
        public void SlideIn(RectTransform target, SlideDirection direction, float duration = 0.4f, System.Action onComplete = null)
        {
            Vector2 startPos = GetSlideStartPosition(target, direction);
            Vector2 endPos = target.anchoredPosition;
            
            target.anchoredPosition = startPos;
            
            AnimateVector3($"SlideIn_{target.GetInstanceID()}", 
                startPos, endPos, duration,
                (value) => target.anchoredPosition = value,
                onComplete,
                AnimationCurve.EaseInOut(0, 0, 1, 1));
        }
        
        #endregion
        
        #region Helper Methods
        
        private Vector2 GetSlideStartPosition(RectTransform target, SlideDirection direction)
        {
            Vector2 canvasSize = ((RectTransform)target.root).rect.size;
            Vector2 currentPos = target.anchoredPosition;
            
            switch (direction)
            {
                case SlideDirection.Left:
                    return new Vector2(-canvasSize.x, currentPos.y);
                case SlideDirection.Right:
                    return new Vector2(canvasSize.x, currentPos.y);
                case SlideDirection.Top:
                    return new Vector2(currentPos.x, canvasSize.y);
                case SlideDirection.Bottom:
                    return new Vector2(currentPos.x, -canvasSize.y);
                default:
                    return currentPos;
            }
        }
        
        #endregion
        
        public enum SlideDirection
        {
            Left,
            Right,
            Top,
            Bottom
        }
    }
}