using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;

namespace IronManSim.UI.Animations
{
    /// <summary>
    /// Handles all menu transition animations with Iron Man themed effects
    /// </summary>
    public class MenuTransitions : MonoBehaviour
    {
        [System.Serializable]
        public class MenuPanel
        {
            public string menuName;
            public RectTransform panel;
            public CanvasGroup canvasGroup;
            public List<UIElement> elements = new List<UIElement>();
        }
        
        [System.Serializable]
        public class UIElement
        {
            public RectTransform transform;
            public float animationDelay = 0f;
            public AnimationType animationType = AnimationType.FadeScale;
        }
        
        public enum AnimationType
        {
            FadeScale,
            SlideLeft,
            SlideRight,
            SlideTop,
            SlideBottom,
            RotateIn,
            HologramGlitch,
            ArcReactorPulse
        }
        
        public enum TransitionType
        {
            Instant,
            Fade,
            SlideHorizontal,
            SlideVertical,
            IronManAssemble,
            HolographicWipe,
            ReactorBurst
        }
        
        [Header("Menu Configuration")]
        [SerializeField] private List<MenuPanel> menuPanels = new List<MenuPanel>();
        [SerializeField] private MenuPanel currentMenu;
        
        [Header("Transition Settings")]
        [SerializeField] private TransitionType defaultTransition = TransitionType.IronManAssemble;
        [SerializeField] private float transitionDuration = 0.5f;
        [SerializeField] private AnimationCurve transitionCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
        
        [Header("Effects")]
        [SerializeField] private GameObject hologramLinesPrefab;
        [SerializeField] private GameObject arcReactorEffectPrefab;
        [SerializeField] private Material hologramMaterial;
        [SerializeField] private Color ironManRed = new Color(0.8f, 0.1f, 0.1f);
        [SerializeField] private Color arcReactorBlue = new Color(0.2f, 0.8f, 1f);
        
        [Header("Audio")]
        [SerializeField] private AudioSource audioSource;
        [SerializeField] private AudioClip assembleSound;
        [SerializeField] private AudioClip hologramSound;
        [SerializeField] private AudioClip reactorSound;
        
        private UIAnimationController animController;
        private Dictionary<string, MenuPanel> menuLookup = new Dictionary<string, MenuPanel>();
        private Coroutine currentTransition;
        
        void Start()
        {
            animController = UIAnimationController.Instance;
            
            // Build menu lookup
            foreach (var menu in menuPanels)
            {
                menuLookup[menu.menuName] = menu;
                
                // Ensure canvas groups exist
                if (menu.canvasGroup == null)
                {
                    menu.canvasGroup = menu.panel.GetComponent<CanvasGroup>();
                    if (menu.canvasGroup == null)
                    {
                        menu.canvasGroup = menu.panel.gameObject.AddComponent<CanvasGroup>();
                    }
                }
                
                // Hide all menus except current
                if (menu != currentMenu)
                {
                    menu.panel.gameObject.SetActive(false);
                }
            }
        }
        
        #region Public Methods
        
        /// <summary>
        /// Transition to a specific menu
        /// </summary>
        public void TransitionToMenu(string menuName, TransitionType? transitionType = null)
        {
            if (!menuLookup.ContainsKey(menuName))
            {
                Debug.LogWarning($"Menu '{menuName}' not found!");
                return;
            }
            
            MenuPanel targetMenu = menuLookup[menuName];
            
            if (targetMenu == currentMenu)
            {
                return;
            }
            
            if (currentTransition != null)
            {
                StopCoroutine(currentTransition);
            }
            
            TransitionType transition = transitionType ?? defaultTransition;
            currentTransition = StartCoroutine(PerformTransition(currentMenu, targetMenu, transition));
        }
        
        /// <summary>
        /// Show menu with animation
        /// </summary>
        public void ShowMenu(string menuName, AnimationType animationType = AnimationType.FadeScale)
        {
            if (!menuLookup.ContainsKey(menuName))
            {
                return;
            }
            
            MenuPanel menu = menuLookup[menuName];
            StartCoroutine(AnimateMenuIn(menu, animationType));
        }
        
        /// <summary>
        /// Hide menu with animation
        /// </summary>
        public void HideMenu(string menuName, AnimationType animationType = AnimationType.FadeScale)
        {
            if (!menuLookup.ContainsKey(menuName))
            {
                return;
            }
            
            MenuPanel menu = menuLookup[menuName];
            StartCoroutine(AnimateMenuOut(menu, animationType));
        }
        
        #endregion
        
        #region Transition Implementations
        
        private IEnumerator PerformTransition(MenuPanel fromMenu, MenuPanel toMenu, TransitionType transition)
        {
            switch (transition)
            {
                case TransitionType.Instant:
                    yield return InstantTransition(fromMenu, toMenu);
                    break;
                    
                case TransitionType.Fade:
                    yield return FadeTransition(fromMenu, toMenu);
                    break;
                    
                case TransitionType.SlideHorizontal:
                    yield return SlideTransition(fromMenu, toMenu, true);
                    break;
                    
                case TransitionType.SlideVertical:
                    yield return SlideTransition(fromMenu, toMenu, false);
                    break;
                    
                case TransitionType.IronManAssemble:
                    yield return IronManAssembleTransition(fromMenu, toMenu);
                    break;
                    
                case TransitionType.HolographicWipe:
                    yield return HolographicWipeTransition(fromMenu, toMenu);
                    break;
                    
                case TransitionType.ReactorBurst:
                    yield return ReactorBurstTransition(fromMenu, toMenu);
                    break;
            }
            
            currentMenu = toMenu;
            currentTransition = null;
        }
        
        private IEnumerator InstantTransition(MenuPanel fromMenu, MenuPanel toMenu)
        {
            if (fromMenu != null)
            {
                fromMenu.panel.gameObject.SetActive(false);
            }
            
            toMenu.panel.gameObject.SetActive(true);
            toMenu.canvasGroup.alpha = 1;
            
            yield return null;
        }
        
        private IEnumerator FadeTransition(MenuPanel fromMenu, MenuPanel toMenu)
        {
            toMenu.panel.gameObject.SetActive(true);
            toMenu.canvasGroup.alpha = 0;
            
            // Fade out old menu
            if (fromMenu != null)
            {
                animController.FadeOut(fromMenu.canvasGroup, transitionDuration * 0.5f);
                yield return new WaitForSeconds(transitionDuration * 0.5f);
                fromMenu.panel.gameObject.SetActive(false);
            }
            
            // Fade in new menu
            animController.FadeIn(toMenu.canvasGroup, transitionDuration * 0.5f);
            yield return new WaitForSeconds(transitionDuration * 0.5f);
        }
        
        private IEnumerator SlideTransition(MenuPanel fromMenu, MenuPanel toMenu, bool horizontal)
        {
            toMenu.panel.gameObject.SetActive(true);
            
            Vector2 canvasSize = ((RectTransform)transform.root).rect.size;
            Vector2 slideOffset = horizontal ? 
                new Vector2(canvasSize.x, 0) : 
                new Vector2(0, canvasSize.y);
            
            // Position new menu off-screen
            toMenu.panel.anchoredPosition = slideOffset;
            toMenu.canvasGroup.alpha = 1;
            
            // Slide both menus
            if (fromMenu != null)
            {
                animController.AnimateVector3($"SlideOut_{fromMenu.menuName}",
                    fromMenu.panel.anchoredPosition,
                    -slideOffset,
                    transitionDuration,
                    (pos) => fromMenu.panel.anchoredPosition = pos,
                    () => fromMenu.panel.gameObject.SetActive(false));
            }
            
            animController.AnimateVector3($"SlideIn_{toMenu.menuName}",
                toMenu.panel.anchoredPosition,
                Vector2.zero,
                transitionDuration,
                (pos) => toMenu.panel.anchoredPosition = pos);
            
            yield return new WaitForSeconds(transitionDuration);
        }
        
        private IEnumerator IronManAssembleTransition(MenuPanel fromMenu, MenuPanel toMenu)
        {
            PlaySound(assembleSound);
            
            // Hide old menu with disassemble effect
            if (fromMenu != null)
            {
                yield return DisassembleMenu(fromMenu);
                fromMenu.panel.gameObject.SetActive(false);
            }
            
            // Show new menu with assemble effect
            toMenu.panel.gameObject.SetActive(true);
            yield return AssembleMenu(toMenu);
        }
        
        private IEnumerator AssembleMenu(MenuPanel menu)
        {
            menu.canvasGroup.alpha = 1;
            
            // Animate each element
            foreach (var element in menu.elements)
            {
                PrepareElementForAnimation(element);
                StartCoroutine(AnimateElementIn(element));
            }
            
            // Main panel scale animation
            menu.panel.localScale = Vector3.zero;
            animController.AnimateVector3($"AssembleScale_{menu.menuName}",
                Vector3.zero,
                Vector3.one,
                transitionDuration,
                (scale) => menu.panel.localScale = scale,
                null,
                AnimationCurve.EaseInOut(0, 0, 1, 1));
            
            yield return new WaitForSeconds(transitionDuration);
        }
        
        private IEnumerator DisassembleMenu(MenuPanel menu)
        {
            // Animate each element out
            foreach (var element in menu.elements)
            {
                StartCoroutine(AnimateElementOut(element));
            }
            
            // Main panel scale animation
            animController.AnimateVector3($"DisassembleScale_{menu.menuName}",
                menu.panel.localScale,
                Vector3.zero,
                transitionDuration * 0.8f,
                (scale) => menu.panel.localScale = scale);
            
            yield return new WaitForSeconds(transitionDuration * 0.8f);
        }
        
        private IEnumerator HolographicWipeTransition(MenuPanel fromMenu, MenuPanel toMenu)
        {
            PlaySound(hologramSound);
            
            // Create holographic wipe effect
            if (hologramLinesPrefab != null)
            {
                GameObject wipeEffect = Instantiate(hologramLinesPrefab, transform);
                RectTransform wipeRect = wipeEffect.GetComponent<RectTransform>();
                wipeRect.anchorMin = Vector2.zero;
                wipeRect.anchorMax = Vector2.one;
                wipeRect.sizeDelta = Vector2.zero;
                wipeRect.anchoredPosition = Vector2.zero;
                
                // Animate wipe across screen
                Image wipeImage = wipeEffect.GetComponent<Image>();
                if (wipeImage != null && hologramMaterial != null)
                {
                    wipeImage.material = hologramMaterial;
                }
                
                float wipeProgress = 0;
                while (wipeProgress < 1)
                {
                    wipeProgress += Time.deltaTime / transitionDuration;
                    
                    // Update shader properties
                    if (hologramMaterial != null)
                    {
                        hologramMaterial.SetFloat("_WipeProgress", wipeProgress);
                    }
                    
                    // Switch menus at halfway point
                    if (wipeProgress >= 0.5f && fromMenu != null && fromMenu.panel.gameObject.activeSelf)
                    {
                        fromMenu.panel.gameObject.SetActive(false);
                        toMenu.panel.gameObject.SetActive(true);
                        toMenu.canvasGroup.alpha = 1;
                    }
                    
                    yield return null;
                }
                
                Destroy(wipeEffect);
            }
            else
            {
                // Fallback to fade
                yield return FadeTransition(fromMenu, toMenu);
            }
        }
        
        private IEnumerator ReactorBurstTransition(MenuPanel fromMenu, MenuPanel toMenu)
        {
            PlaySound(reactorSound);
            
            // Create arc reactor burst effect
            if (arcReactorEffectPrefab != null)
            {
                GameObject burst = Instantiate(arcReactorEffectPrefab, transform);
                RectTransform burstRect = burst.GetComponent<RectTransform>();
                burstRect.anchoredPosition = Vector2.zero;
                
                // Animate burst
                burstRect.localScale = Vector3.zero;
                animController.AnimateVector3($"ReactorBurst",
                    Vector3.zero,
                    Vector3.one * 3,
                    transitionDuration,
                    (scale) => 
                    {
                        burstRect.localScale = scale;
                        
                        // Fade out burst
                        Image burstImage = burst.GetComponent<Image>();
                        if (burstImage != null)
                        {
                            Color color = burstImage.color;
                            color.a = 1f - (scale.x / 3f);
                            burstImage.color = color;
                        }
                    },
                    () => Destroy(burst));
            }
            
            // Flash transition menus
            yield return new WaitForSeconds(transitionDuration * 0.3f);
            
            if (fromMenu != null)
            {
                fromMenu.canvasGroup.alpha = 0;
                fromMenu.panel.gameObject.SetActive(false);
            }
            
            toMenu.panel.gameObject.SetActive(true);
            toMenu.canvasGroup.alpha = 0;
            animController.FadeIn(toMenu.canvasGroup, transitionDuration * 0.5f);
            
            yield return new WaitForSeconds(transitionDuration * 0.7f);
        }
        
        #endregion
        
        #region Element Animations
        
        private void PrepareElementForAnimation(UIElement element)
        {
            switch (element.animationType)
            {
                case AnimationType.FadeScale:
                    element.transform.localScale = Vector3.zero;
                    break;
                    
                case AnimationType.SlideLeft:
                    element.transform.anchoredPosition += new Vector2(-200, 0);
                    break;
                    
                case AnimationType.SlideRight:
                    element.transform.anchoredPosition += new Vector2(200, 0);
                    break;
                    
                case AnimationType.SlideTop:
                    element.transform.anchoredPosition += new Vector2(0, 200);
                    break;
                    
                case AnimationType.SlideBottom:
                    element.transform.anchoredPosition += new Vector2(0, -200);
                    break;
                    
                case AnimationType.RotateIn:
                    element.transform.localRotation = Quaternion.Euler(0, 0, 90);
                    element.transform.localScale = Vector3.zero;
                    break;
            }
        }
        
        private IEnumerator AnimateElementIn(UIElement element)
        {
            yield return new WaitForSeconds(element.animationDelay);
            
            Vector3 targetPosition = element.transform.anchoredPosition;
            Vector3 targetScale = Vector3.one;
            Quaternion targetRotation = Quaternion.identity;
            
            switch (element.animationType)
            {
                case AnimationType.SlideLeft:
                    targetPosition = element.transform.anchoredPosition + new Vector2(200, 0);
                    break;
                case AnimationType.SlideRight:
                    targetPosition = element.transform.anchoredPosition - new Vector2(200, 0);
                    break;
                case AnimationType.SlideTop:
                    targetPosition = element.transform.anchoredPosition - new Vector2(0, 200);
                    break;
                case AnimationType.SlideBottom:
                    targetPosition = element.transform.anchoredPosition + new Vector2(0, 200);
                    break;
            }
            
            float duration = 0.3f;
            string animId = $"ElementIn_{element.transform.GetInstanceID()}";
            
            // Position animation
            if (element.animationType == AnimationType.SlideLeft || 
                element.animationType == AnimationType.SlideRight ||
                element.animationType == AnimationType.SlideTop ||
                element.animationType == AnimationType.SlideBottom)
            {
                animController.AnimateVector3(animId + "_pos",
                    element.transform.anchoredPosition,
                    targetPosition,
                    duration,
                    (pos) => element.transform.anchoredPosition = pos);
            }
            
            // Scale animation
            if (element.animationType == AnimationType.FadeScale ||
                element.animationType == AnimationType.RotateIn)
            {
                animController.AnimateVector3(animId + "_scale",
                    element.transform.localScale,
                    targetScale,
                    duration,
                    (scale) => element.transform.localScale = scale);
            }
            
            // Rotation animation
            if (element.animationType == AnimationType.RotateIn)
            {
                animController.AnimateFloat(animId + "_rot",
                    90f,
                    0f,
                    duration,
                    (angle) => element.transform.localRotation = Quaternion.Euler(0, 0, angle));
            }
        }
        
        private IEnumerator AnimateElementOut(UIElement element)
        {
            float duration = 0.2f;
            string animId = $"ElementOut_{element.transform.GetInstanceID()}";
            
            animController.AnimateVector3(animId,
                element.transform.localScale,
                Vector3.zero,
                duration,
                (scale) => element.transform.localScale = scale);
            
            yield return new WaitForSeconds(duration);
        }
        
        #endregion
        
        #region Menu In/Out Animations
        
        private IEnumerator AnimateMenuIn(MenuPanel menu, AnimationType animationType)
        {
            menu.panel.gameObject.SetActive(true);
            menu.canvasGroup.alpha = 1;
            
            foreach (var element in menu.elements)
            {
                element.animationType = animationType;
                PrepareElementForAnimation(element);
                StartCoroutine(AnimateElementIn(element));
            }
            
            yield return new WaitForSeconds(transitionDuration);
        }
        
        private IEnumerator AnimateMenuOut(MenuPanel menu, AnimationType animationType)
        {
            foreach (var element in menu.elements)
            {
                StartCoroutine(AnimateElementOut(element));
            }
            
            yield return new WaitForSeconds(transitionDuration * 0.8f);
            
            menu.panel.gameObject.SetActive(false);
        }
        
        #endregion
        
        #region Utility Methods
        
        private void PlaySound(AudioClip clip)
        {
            if (audioSource != null && clip != null)
            {
                audioSource.PlayOneShot(clip);
            }
        }
        
        #endregion
    }
}