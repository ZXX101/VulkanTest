#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>//GLFW will include its own definitions and automatically load the Vulkan header with it

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <limits>
#include <algorithm>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

#include <fstream>
//ate: Start reading at the end of the file
//binary: Read the file as binary file(avoid text transformations)
static std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	//The advantage of starting to read at the end of the file is that we can use the read position to determine the size of the file and allocate a buffer
	//从结尾读可以方便知道文件大小，用于分配vector
	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);//not create OpenGL context
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);// disable window resize for now
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}

	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();//call this createXXX func after create Instance
		pickPyhsicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createGraphicsPipeline();
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	//explicity destryo 
	void cleanup() {

		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}


		vkDestroySwapchainKHR(device, swapChain, nullptr);
		vkDestroyDevice(device, nullptr);
		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
		vkDestroySurfaceKHR(instance, surface, nullptr);//call this destroyXXX func before Instance Destroy
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

private:
	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requsted,but not available");
		}
		VkApplicationInfo appInfo{};//leave pNext as nullptr,because it needs to be explicit specify
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;



		//#these are intergrated in func getRequiredExtensions()
		//use glfw to interface with the window sys(window system, not windows system)
		//uint32_t glfwExtensionCount = 0;
		//const char** glfwExtensions;
		//glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		//createInfo.enabledExtensionCount = glfwExtensionCount;
		//createInfo.ppEnabledExtensionNames = glfwExtensions;

		//#replace above code
		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();


		//#Release:not use validation layers
		//#Debug:use validation layers
#ifdef NDEBUG
		createInfo.enabledLayerCount = 0;//two layersXX param determine the validation layer,leave them empty
#else
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();//

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else {
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}
#endif


		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance");
		}



		//check for extension support
		uint32_t extensionCount = 0;
		//request just the number of extensions by leaving the latter parameter empty
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> allextensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, allextensions.data());//vector.data() supply the pointer to access the lowlayer array of vector

		std::cout << "available extensions:\n";

		for (const auto& extension : allextensions) {
			std::cout << "\t" << extension.extensionName << '\n';
		}
	}

	//## validation layer dependence Funcs
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = (PFN_vkDebugUtilsMessengerCallbackEXT)debugCallback;
	}


	void setupDebugMessenger() {
		if (!enableValidationLayers) return;
		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		/*
		check this func's valid usage online
		second param is null:,it set first param to support layer num
		else ,it will get layers point and set to second param by the num of first param */
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerProperties.layerName, layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}
		return true;
	}

	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;//存储指针数组的起始地址的指针
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		//使用另一个数组（vector或普通数组）的指针来初始化，参数为begin & end pointer
		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);//use marco to avoid typos
		}

		return extensions;

	}

	static  VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageServerity,//严重性 can use compare
		VkDebugUtilsMessageTypeFlagsEXT messageType,//无关规范或性能，规范相关或可能的错误，性能相关
		/*
		pMessage: The debug message as a null-terminated string

		pObjects: Array of Vulkan object handles related to the message

		objectCount: Number of objects in array*/
		VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,//
		void* puserData) {
		std::cerr << "validation layer" << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

	//## Physical devices and queue family dependence Funcs
	void pickPyhsicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support");
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		//use a set of strings to represent the unconfirmed required extension
		//then we can easily tick off them while enumerating
		std::set<std::string> requiredExtension(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtension.erase(extension.extensionName);
		}
		return requiredExtension.empty();
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		//VkPhysicalDeviceProperties deviceProperties;
		//VkPhysicalDeviceFeatures deviceFeatures;
		//vkGetPhysicalDeviceProperties(device, &deviceProperties);
		//vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
		//return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader;
		//return true;
		QueueFamilyIndices indices = findQueueFamilies(device);
		bool extensionsSupport = checkDeviceExtensionSupport(device);
		bool swapChainAdequate = false;
		if (extensionsSupport) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}
		return indices.isComplete() && extensionsSupport && swapChainAdequate;
	}

	/// @brief std::optional is a wrapper that contains no value until you assign something to it. At any point you can query if it contains a value or not by calling its has_value() member function
	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete()const {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	/// @brief  almost every operation in Vulkan, anything from drawing to uploading textures, requires commands to be submitted to a queue.
	/// There are different types of queues that originate from different queue families and each family of queues allows only a subset of commands
	/// For example, there could be a queue family that only allows processing of compute commands or one that only allows memory transfer related commands.
	/// @param device 
	/// @return 
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			// what is VK_QUEUE_GRAPHICS_BIT
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if (presentSupport) {
				indices.presentFamily = i;
			}
			if (indices.isComplete()) {//early break
				break;
			}
			i++;
		}

		return indices;
	}

	//##Logical device and queues
	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		//presentation queue
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(),indices.presentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}
		//////be replaced by above
		////device queue  
		//VkDeviceQueueCreateInfo queueCreateInfo{};
		//queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		//queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
		//queueCreateInfo.queueCount = 1;

		//Vulkan lets you assign priorities to queues to 
		// influence the scheduling of command buffer execution 
		// using floating point numbers between 0.0 and 1.0. 
		// This is required even if there is only a single queue:
		//queueCreateInfo.pQueuePriorities = &queuePriority;
		//////be replaced END


		//features
		VkPhysicalDeviceFeatures deviceFeatures{};

		//fill the main VkDeviceCreateInfo with previous two structures
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;//this enum value similar to class type
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();

		createInfo.pEnabledFeatures = &deviceFeatures;

		//enable device extensions
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	//swap chain
	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		//capabilities
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		//formats
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		//present mode
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}


		return details;
	}

	//find the right settings：
	//	* Surface format(color depth)
	//
	//	* Presentation mode(conditions for "swapping" images to the screen)
	//
	//	* Swap extent(resolution of images in swap chain)
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	//consider energy-usage,use fifo in Mobile Device.
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;//proves it already has value
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent{//extent 有大小、长度的意思
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};
			//clamp :按参数2,3范围截取参数1
			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);//most about Resolution of iamges which are going to be present in display

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		//0 代表没限制，所以判断大于0才知道最大值有值，此时imageCount等于最小值，如果大于最大值，那么最小值就大于最大值了，但是不能超过最大值，所以把最大值赋值给imageCount，这里很奇怪，为什么最小值会大于最大值
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		//Create Info
		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(),indices.presentFamily.value() };

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;//并行，无需传递图像
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;//独立的
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;//like a 90 degree clockwise rotation or horizontal flip,To specify that you do not want any transformation, simply specify the current transformation.
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;//if the alpha channel should be used for blending with other windows in the window system
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;//设置为真，则例如被其他窗口遮蔽的部分就不关心。可以提高性能。除非需要读取被遮蔽的像素和预测的结果
		createInfo.oldSwapchain = VK_NULL_HANDLE;//例如窗口缩放时，必须从头开始重新创建swapchain并且给这个参数指定旧的swapchain。现在简化处理我们只用一个，不要缩放窗口。

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		//store images
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}


	//windows surface
	//use glfwCreateWindowSurface instead of xxxWIN32XXX funcs to avoid platform specifiy
	//there is none xxxWIN32xxx funcs in tutroial,notice this.
	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	//image views
	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];

			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;

			//The components field allows you to swizzle the color channels around.
			//For example, you can map all of the channels to the red channel for a monochrome texture.
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			//The subresourceRange field describes what the image’s purpose is and which part of the image should be accessed. 
			// Our images will be used as color targets without any mipmapping levels or multiple layers.
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			//
			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}


		}
	}

	//Graphics Pipeline
	//
	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("shaders/vert.spv");
		auto fragShaderCode = readFile("shaders/frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		//shader stage creation
		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		//vertex input, no vertex data to load for now
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

		//Input Assembly
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		//如果在dynamic state里设置了动态使用viewport和scissor，就不用手动设置，只要指定数量即可
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		////viewports
		//VkViewport viewport{};
		//viewport.x = 0.0f;
		//viewport.y = 0.0f;
		//viewport.width = (float)swapChainExtent.width;
		//viewport.height = (float)swapChainExtent.height;
		//viewport.minDepth = 0.0f;
		//viewport.maxDepth = 1.0f;
		////scissor rectangle 不能超过viewport
		//VkRect2D scissor{};
		//scissor.offset = { 0, 0 };
		//scissor.extent = swapChainExtent;

		//深度测试、面剔除、scissor test裁剪矩形测试、输出整个几何体或只是线（线框渲染）
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;//环绕顺序为顺时针则为正面
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; // Optional
		rasterizer.depthBiasClamp = 0.0f; // Optional
		rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

		//dynamic state
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();



		//clean up
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	//Before we can pass the code to the pipeline, we have to wrap it in a VkShaderModule object.
	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}
		return shaderModule;
	}
private:
	GLFWwindow* window;
	VkInstance instance;
	//validation layers
	VkDebugUtilsMessengerEXT debugMessenger;

	//window surface
	VkSurfaceKHR surface;

	//physical device
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	//Logical Device
	VkDevice device;
	//queue handles
	VkQueue graphicsQueue;//implicity destroyed up when the device is destroyed
	VkQueue presentQueue;//

	//swap chain
	VkSwapchainKHR swapChain;
	//store the handles of vkimages in swapchain
	std::vector<VkImage> swapChainImages;//automatically cleaned up once that swap chain has been destroyed.
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	//image views
	std::vector<VkImageView> swapChainImageViews;
};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}