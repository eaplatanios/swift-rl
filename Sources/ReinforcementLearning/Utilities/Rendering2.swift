// // Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License"); you may not
// // use this file except in compliance with the License. You may obtain a copy of
// // the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// // WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// // License for the specific language governing permissions and limitations under
// // the License.

// import CGLFW
// import CVulkan

// @inlinable
// internal func withCStrings<R>(
//   _ args: [String],
//   _ body: (UnsafePointer<UnsafePointer<Int8>?>?) -> R
// ) -> R {
//   var cStrings = args.map { UnsafePointer<Int8>(strdup($0)) }
//   defer { cStrings.forEach { free(UnsafeMutablePointer(mutating: $0)) } }
//   return body(&cStrings)
// }

// public class RenderingWindow {
//   public let name: String

//   public let applicationVersion: UInt32 = 1
//   public let vulkanEngineVersion: UInt32 = 0
//   public let vulkanAPIVersion: UInt32 = 0

//   public let vulkanEnabledLayers: [String] = []
//   public let vulkanDeviceScoringFunction: (VulkanPhysicalDevice) -> Int? =
//     VulkanPhysicalDevice.defaultScore(for:)

//   @usableFromInline internal let width: Int
//   @usableFromInline internal let height: Int
//   @usableFromInline internal let framesPerSecond: Double?
//   @usableFromInline internal var window: OpaquePointer? = nil
//   @usableFromInline internal var isOpen: Bool = true

//   @usableFromInline internal var vulkanInstance: VkInstance? = nil
//   @usableFromInline internal var vulkanPhysicalDevice: VulkanPhysicalDevice? = nil

//   @inlinable
//   public init(name: String, width: Int, height: Int, framesPerSecond: Double? = nil) throws {
//     self.name = name
//     self.width = width
//     self.height = height
//     self.framesPerSecond = framesPerSecond
//     try! glfwInit()
//     try! vulkanInit()
//   }

//   @inlinable
//   deinit {
//     close()
//   }

//   @inlinable
//   internal func glfwInit() throws {
//     // Initialize GLFW.
//     if CGLFW.glfwInit() == 0 { throw ReinforcementLearningError.renderingError("Failed to initialize GLFW.") }

//     // Since we are using Vulkan, we need to tell GLFW to not create an OpenGL context.
//     glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API)
//     glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE)

//     // Open a new window.
//     guard let window = glfwCreateWindow(Int32(width), Int32(height), name, nil, nil) else {
//       glfwTerminate()
//       throw ReinforcementLearningError.renderingError("Failed to open a GLFW window.")
//     }

//     self.window = window
//     glfwMakeContextCurrent(window)
//   }

//   @inlinable
//   internal func vulkanInit() throws {
//     try! vulkanCreateInstance()
//     try! vulkanPickPhysicalDevice()
//     try! vulkanCreateLogicalDevice()
//   }

//   @inlinable
//   internal func vulkanCreateInstance() throws {
//     var applicationInformation = VkApplicationInfo(
//       sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
//       pNext: nil,
//       pApplicationName: name,
//       applicationVersion: applicationVersion,
//       pEngineName: nil,
//       engineVersion: vulkanEngineVersion,
//       apiVersion: vulkanAPIVersion)

//     // Get the required GLFW extensions.
//     var glfwExtensionCount: UInt32 = 0
//     let glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount)

//     var information = withCStrings(vulkanEnabledLayers) { enabledLayers in
//       VkInstanceCreateInfo(
//         sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
//         pNext: nil,
//         flags: 0,
//         pApplicationInfo: &applicationInformation,
//         enabledLayerCount: UInt32(self.vulkanEnabledLayers.count),
//         ppEnabledLayerNames: enabledLayers,
//         enabledExtensionCount: glfwExtensionCount,
//         ppEnabledExtensionNames: glfwExtensions)
//     }

//     let result = vkCreateInstance(&information, nil, &vulkanInstance)
//     if result != VK_SUCCESS {
//       throw ReinforcementLearningError.renderingError(
//         "Failed to create a Vulkan instance. Error code: \(String(describing: result.rawValue))")
//     }
//   }

//   @inlinable
//   internal func vulkanPickPhysicalDevice() throws {
//     let devices = vulkanDevices()
//     let scores = devices.map(vulkanDeviceScoringFunction)
//     var maxScore = Int.min
//     var device: VulkanPhysicalDevice? = nil
//     for (currentDevice, currentScore) in zip(devices, scores) {
//       if let s = currentScore, s >= maxScore {
//         maxScore = s
//         device = currentDevice
//       }
//     }
//     if let d = device {
//       vulkanPhysicalDevice = d
//     } else {
//       throw ReinforcementLearningError.renderingError("Failed to find a suitable device for rendering.")
//     }
//   }

//   @inlinable
//   internal func vulkanCreateLogicalDevice() throws {
    
//   }

//   @inlinable
//   public func vulkanExtensions() -> [String] {
//     var extensionCount: UInt32 = 0
//     vkEnumerateInstanceExtensionProperties(nil, &extensionCount, nil)
//     let extensionsPointer = UnsafeMutablePointer<VkExtensionProperties>.allocate(
//       capacity: Int(extensionCount))
//     defer { extensionsPointer.deallocate() }
//     vkEnumerateInstanceExtensionProperties(nil, &extensionCount, extensionsPointer)
//     let extensionsBufferPointer = UnsafeBufferPointer(
//       start: extensionsPointer,
//       count: Int(extensionCount))
//     return [VkExtensionProperties](extensionsBufferPointer).map {
//       var name = $0.extensionName
//       return withUnsafeBytes(of: &name) { rawPointer -> String in
//         String(cString: rawPointer.baseAddress!.assumingMemoryBound(to: CChar.self))
//       }
//     }
//   }

//   @inlinable
//   public func vulkanDevices() -> [VulkanPhysicalDevice] {
//     var deviceCount: UInt32 = 0
//     vkEnumeratePhysicalDevices(vulkanInstance, &deviceCount, nil)
//     if (deviceCount == 0) { return [VulkanPhysicalDevice]() }
//     let devicesPointer = UnsafeMutablePointer<VkPhysicalDevice?>.allocate(
//       capacity: Int(deviceCount))
//     defer { devicesPointer.deallocate() }
//     vkEnumeratePhysicalDevices(vulkanInstance, &deviceCount, devicesPointer)
//     let devicesBufferPointer = UnsafeBufferPointer(start: devicesPointer, count: Int(deviceCount))
//     return [VkPhysicalDevice?](devicesBufferPointer).map { VulkanPhysicalDevice(device: $0!) }
//   }

//   @inlinable
//   public func close() {
//     if let vulkanInstance = self.vulkanInstance {
//       vkDestroyInstance(vulkanInstance, nil)
//       self.vulkanInstance = nil
//     }
//     if let window = self.window {
//       glfwDestroyWindow(window)
//       glfwTerminate()
//       self.isOpen = false
//       self.window = nil
//     }
//   }
// }

// public struct VulkanPhysicalDevice {
//   @usableFromInline internal let device: VkPhysicalDevice

//   public let name: String
//   public let apiVersion: UInt32
//   public let driverVersion: UInt32
//   public let vendorID: UInt32
//   public let deviceID: UInt32
//   public let type: RenderingDeviceType
//   public let limits: VkPhysicalDeviceLimits
//   public let sparseProperties: VkPhysicalDeviceSparseProperties
//   public let features: VkPhysicalDeviceFeatures

//   @inlinable
//   internal init(device: VkPhysicalDevice) {
//     let propertiesPointer = UnsafeMutablePointer<VkPhysicalDeviceProperties>.allocate(capacity: 1)
//     vkGetPhysicalDeviceProperties(device, propertiesPointer)
//     let properties = propertiesPointer.pointee
//     self.device = device
//     var deviceName = properties.deviceName
//     self.name = withUnsafeBytes(of: &deviceName) { rawPointer -> String in
//       String(cString: rawPointer.baseAddress!.assumingMemoryBound(to: CChar.self))
//     }
//     self.apiVersion = properties.apiVersion
//     self.driverVersion = properties.driverVersion
//     self.vendorID = properties.vendorID
//     self.deviceID = properties.deviceID
//     self.type = RenderingDeviceType(from: properties.deviceType)
//     self.limits = properties.limits
//     self.sparseProperties = properties.sparseProperties
//     let featuresPointer = UnsafeMutablePointer<VkPhysicalDeviceFeatures>.allocate(capacity: 1)
//     vkGetPhysicalDeviceFeatures(device, featuresPointer)
//     self.features = featuresPointer.pointee
//   }
// }

// extension VulkanPhysicalDevice {
//   @inlinable
//   public var queueFamilies: [VulkanQueueFamily] {
//     var queueFamilyCount: UInt32 = 0
//     vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nil)
//     let queueFamiliesPoiner = UnsafeMutablePointer<VkQueueFamilyProperties>.allocate(
//       capacity: Int(queueFamilyCount))
//     vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamiliesPoiner)
//     let queueFamiliesBufferPointer = UnsafeBufferPointer(
//       start: queueFamiliesPoiner,
//       count: Int(queueFamilyCount))
//     return queueFamiliesBufferPointer.enumerated().map {
//       VulkanQueueFamily(at: $0, withProperties: $1)
//     }
//   }
// }

// extension VulkanPhysicalDevice {
//   public static func defaultScore(for device: VulkanPhysicalDevice) -> Int? {
//     // if device.features.geometryShader == 0 { return 0 }
//     if !device.queueFamilies.contains(where: { $0.supportsGraphics }) { return nil }
//     var score = 0
//     if device.type == .discreteGPU { score += 1000 }
//     score += Int(device.limits.maxImageDimension2D)
//     return score
//   }
// }

// public struct VulkanQueueFamily {
//   public let index: Int
//   public let supportsGraphics: Bool
//   public let supportsCompute: Bool
//   public let supportsTransfer: Bool
//   public let supportsSparseBinding: Bool
//   public let supportsProtected: Bool
//   public let queueCount: UInt32
//   public let timestampValidBits: UInt32
//   public let minImageTransferGranularity: VulkanExtent3D

//   @inlinable
//   internal init(at index: Int, withProperties properties: VkQueueFamilyProperties) {
//     self.index = index
//     self.supportsGraphics = properties.queueFlags & VK_QUEUE_GRAPHICS_BIT.rawValue != 0
//     self.supportsCompute = properties.queueFlags & VK_QUEUE_COMPUTE_BIT.rawValue != 0
//     self.supportsTransfer = properties.queueFlags & VK_QUEUE_TRANSFER_BIT.rawValue != 0
//     self.supportsSparseBinding = properties.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT.rawValue != 0
//     self.supportsProtected = properties.queueFlags & VK_QUEUE_PROTECTED_BIT.rawValue != 0
//     self.queueCount = properties.queueCount
//     self.timestampValidBits = properties.timestampValidBits
//     self.minImageTransferGranularity = VulkanExtent3D(
//       width: properties.minImageTransferGranularity.width,
//       height: properties.minImageTransferGranularity.height,
//       depth: properties.minImageTransferGranularity.depth)
//   }
// }

// public struct VulkanExtent3D {
//   public let width: UInt32
//   public let height: UInt32
//   public let depth: UInt32

//   @inlinable
//   internal init(width: UInt32, height: UInt32, depth: UInt32) {
//     self.width = width
//     self.height = height
//     self.depth = depth
//   }
// }

// public enum RenderingDeviceType {
//   case integratedGPU, discreteGPU, virtualGPU, cpu, other

//   @inlinable
//   internal init(from vulkanValue: VkPhysicalDeviceType) {
//     switch vulkanValue {
//     case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: self = .integratedGPU
//     case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: self = .discreteGPU
//     case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: self = .virtualGPU
//     case VK_PHYSICAL_DEVICE_TYPE_CPU: self = .cpu
//     case _: self = .other
//     }
//   }
// }
