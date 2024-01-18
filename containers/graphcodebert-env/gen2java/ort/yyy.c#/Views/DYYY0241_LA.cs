// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: DYYY0241_LA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:07
//
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// using Statements
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
using System;
using com.ca.gen.vwrt;
using com.ca.gen.vwrt.types;
using com.ca.gen.vwrt.vdf;
using com.ca.gen.csu.exception;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// START OF LOCAL VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: DYYY0241_LA
  /// </summary>
  [Serializable]
  public class DYYY0241_LA : ViewBase, ILocalView
  {
    private static DYYY0241_LA[] freeArray = new DYYY0241_LA[30];
    private static int countFree = 0;
    
    // Entity View: LOC
    //        Type: DONT_CHANGE_RETURN_CODES
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReturnCodesQ1Ok
    /// </summary>
    private char _LocDontChangeReturnCodesQ1Ok_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReturnCodesQ1Ok
    /// </summary>
    public char LocDontChangeReturnCodesQ1Ok_AS {
      get {
        return(_LocDontChangeReturnCodesQ1Ok_AS);
      }
      set {
        _LocDontChangeReturnCodesQ1Ok_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReturnCodesQ1Ok
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReturnCodesQ1Ok;
    /// <summary>
    /// Attribute for: LocDontChangeReturnCodesQ1Ok
    /// </summary>
    public int LocDontChangeReturnCodesQ1Ok {
      get {
        return(_LocDontChangeReturnCodesQ1Ok);
      }
      set {
        _LocDontChangeReturnCodesQ1Ok = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReturnCodesN10ObjNotFound
    /// </summary>
    private char _LocDontChangeReturnCodesN10ObjNotFound_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReturnCodesN10ObjNotFound
    /// </summary>
    public char LocDontChangeReturnCodesN10ObjNotFound_AS {
      get {
        return(_LocDontChangeReturnCodesN10ObjNotFound_AS);
      }
      set {
        _LocDontChangeReturnCodesN10ObjNotFound_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReturnCodesN10ObjNotFound
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReturnCodesN10ObjNotFound;
    /// <summary>
    /// Attribute for: LocDontChangeReturnCodesN10ObjNotFound
    /// </summary>
    public int LocDontChangeReturnCodesN10ObjNotFound {
      get {
        return(_LocDontChangeReturnCodesN10ObjNotFound);
      }
      set {
        _LocDontChangeReturnCodesN10ObjNotFound = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReturnCodesN43ObjNotDeleted
    /// </summary>
    private char _LocDontChangeReturnCodesN43ObjNotDeleted_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReturnCodesN43ObjNotDeleted
    /// </summary>
    public char LocDontChangeReturnCodesN43ObjNotDeleted_AS {
      get {
        return(_LocDontChangeReturnCodesN43ObjNotDeleted_AS);
      }
      set {
        _LocDontChangeReturnCodesN43ObjNotDeleted_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReturnCodesN43ObjNotDeleted
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReturnCodesN43ObjNotDeleted;
    /// <summary>
    /// Attribute for: LocDontChangeReturnCodesN43ObjNotDeleted
    /// </summary>
    public int LocDontChangeReturnCodesN43ObjNotDeleted {
      get {
        return(_LocDontChangeReturnCodesN43ObjNotDeleted);
      }
      set {
        _LocDontChangeReturnCodesN43ObjNotDeleted = value;
      }
    }
    // Entity View: LOC
    //        Type: DONT_CHANGE_REASON_CODES
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ1Default
    /// </summary>
    private char _LocDontChangeReasonCodesQ1Default_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ1Default
    /// </summary>
    public char LocDontChangeReasonCodesQ1Default_AS {
      get {
        return(_LocDontChangeReasonCodesQ1Default_AS);
      }
      set {
        _LocDontChangeReasonCodesQ1Default_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ1Default
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ1Default;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ1Default
    /// </summary>
    public int LocDontChangeReasonCodesQ1Default {
      get {
        return(_LocDontChangeReasonCodesQ1Default);
      }
      set {
        _LocDontChangeReasonCodesQ1Default = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ121ChildNotFound
    /// </summary>
    private char _LocDontChangeReasonCodesQ121ChildNotFound_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ121ChildNotFound
    /// </summary>
    public char LocDontChangeReasonCodesQ121ChildNotFound_AS {
      get {
        return(_LocDontChangeReasonCodesQ121ChildNotFound_AS);
      }
      set {
        _LocDontChangeReasonCodesQ121ChildNotFound_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ121ChildNotFound
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ121ChildNotFound;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ121ChildNotFound
    /// </summary>
    public int LocDontChangeReasonCodesQ121ChildNotFound {
      get {
        return(_LocDontChangeReasonCodesQ121ChildNotFound);
      }
      set {
        _LocDontChangeReasonCodesQ121ChildNotFound = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ132ChildConcurrencyError
    /// </summary>
    private char _LocDontChangeReasonCodesQ132ChildConcurrencyError_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ132ChildConcurrencyError
    /// </summary>
    public char LocDontChangeReasonCodesQ132ChildConcurrencyError_AS {
      get {
        return(_LocDontChangeReasonCodesQ132ChildConcurrencyError_AS);
      }
      set {
        _LocDontChangeReasonCodesQ132ChildConcurrencyError_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ132ChildConcurrencyError
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ132ChildConcurrencyError;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ132ChildConcurrencyError
    /// </summary>
    public int LocDontChangeReasonCodesQ132ChildConcurrencyError {
      get {
        return(_LocDontChangeReasonCodesQ132ChildConcurrencyError);
      }
      set {
        _LocDontChangeReasonCodesQ132ChildConcurrencyError = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public DYYY0241_LA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public DYYY0241_LA( DYYY0241_LA orig )
    {
      LocDontChangeReturnCodesQ1Ok_AS = orig.LocDontChangeReturnCodesQ1Ok_AS;
      LocDontChangeReturnCodesQ1Ok = orig.LocDontChangeReturnCodesQ1Ok;
      LocDontChangeReturnCodesN10ObjNotFound_AS = orig.LocDontChangeReturnCodesN10ObjNotFound_AS;
      LocDontChangeReturnCodesN10ObjNotFound = orig.LocDontChangeReturnCodesN10ObjNotFound;
      LocDontChangeReturnCodesN43ObjNotDeleted_AS = orig.LocDontChangeReturnCodesN43ObjNotDeleted_AS;
      LocDontChangeReturnCodesN43ObjNotDeleted = orig.LocDontChangeReturnCodesN43ObjNotDeleted;
      LocDontChangeReasonCodesQ1Default_AS = orig.LocDontChangeReasonCodesQ1Default_AS;
      LocDontChangeReasonCodesQ1Default = orig.LocDontChangeReasonCodesQ1Default;
      LocDontChangeReasonCodesQ121ChildNotFound_AS = orig.LocDontChangeReasonCodesQ121ChildNotFound_AS;
      LocDontChangeReasonCodesQ121ChildNotFound = orig.LocDontChangeReasonCodesQ121ChildNotFound;
      LocDontChangeReasonCodesQ132ChildConcurrencyError_AS = orig.LocDontChangeReasonCodesQ132ChildConcurrencyError_AS;
      LocDontChangeReasonCodesQ132ChildConcurrencyError = orig.LocDontChangeReasonCodesQ132ChildConcurrencyError;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static DYYY0241_LA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new DYYY0241_LA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new DYYY0241_LA());
          }
          else 
          {
            DYYY0241_LA result = freeArray[--countFree];
            freeArray[countFree] = null;
            result.Reset(  );
            return(result);
          }
        }
      }
    }
    /// <summary>
    /// Static free instance method
    /// </summary>
    
    public void FreeInstance(  )
    {
      lock (freeArray)
      {
        if ( countFree < freeArray.Length )
        {
          freeArray[countFree++] = this;
        }
      }
    }
    /// <summary>
    /// clone constructor
    /// </summary>
    
    public override Object Clone(  )
    {
      return(new DYYY0241_LA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      LocDontChangeReturnCodesQ1Ok_AS = ' ';
      LocDontChangeReturnCodesQ1Ok = 0;
      LocDontChangeReturnCodesN10ObjNotFound_AS = ' ';
      LocDontChangeReturnCodesN10ObjNotFound = 0;
      LocDontChangeReturnCodesN43ObjNotDeleted_AS = ' ';
      LocDontChangeReturnCodesN43ObjNotDeleted = 0;
      LocDontChangeReasonCodesQ1Default_AS = ' ';
      LocDontChangeReasonCodesQ1Default = 0;
      LocDontChangeReasonCodesQ121ChildNotFound_AS = ' ';
      LocDontChangeReasonCodesQ121ChildNotFound = 0;
      LocDontChangeReasonCodesQ132ChildConcurrencyError_AS = ' ';
      LocDontChangeReasonCodesQ132ChildConcurrencyError = 0;
    }
  }
}
