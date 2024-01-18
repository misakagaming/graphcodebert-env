// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: MYY10121_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:42:01
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
// START OF IMPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: MYY10121_IA
  /// </summary>
  [Serializable]
  public class MYY10121_IA : ViewBase, IImportView
  {
    private static MYY10121_IA[] freeArray = new MYY10121_IA[30];
    private static int countFree = 0;
    
    // Entity View: IMP
    //        Type: IYY1_PARENT
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ParentPinstanceId
    /// </summary>
    private char _ImpIyy1ParentPinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ParentPinstanceId
    /// </summary>
    public char ImpIyy1ParentPinstanceId_AS {
      get {
        return(_ImpIyy1ParentPinstanceId_AS);
      }
      set {
        _ImpIyy1ParentPinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ParentPinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpIyy1ParentPinstanceId;
    /// <summary>
    /// Attribute for: ImpIyy1ParentPinstanceId
    /// </summary>
    public string ImpIyy1ParentPinstanceId {
      get {
        return(_ImpIyy1ParentPinstanceId);
      }
      set {
        _ImpIyy1ParentPinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpIyy1ParentPkeyAttrText
    /// </summary>
    private char _ImpIyy1ParentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpIyy1ParentPkeyAttrText
    /// </summary>
    public char ImpIyy1ParentPkeyAttrText_AS {
      get {
        return(_ImpIyy1ParentPkeyAttrText_AS);
      }
      set {
        _ImpIyy1ParentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpIyy1ParentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpIyy1ParentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpIyy1ParentPkeyAttrText
    /// </summary>
    public string ImpIyy1ParentPkeyAttrText {
      get {
        return(_ImpIyy1ParentPkeyAttrText);
      }
      set {
        _ImpIyy1ParentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public MYY10121_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public MYY10121_IA( MYY10121_IA orig )
    {
      ImpIyy1ParentPinstanceId_AS = orig.ImpIyy1ParentPinstanceId_AS;
      ImpIyy1ParentPinstanceId = orig.ImpIyy1ParentPinstanceId;
      ImpIyy1ParentPkeyAttrText_AS = orig.ImpIyy1ParentPkeyAttrText_AS;
      ImpIyy1ParentPkeyAttrText = orig.ImpIyy1ParentPkeyAttrText;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static MYY10121_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new MYY10121_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new MYY10121_IA());
          }
          else 
          {
            MYY10121_IA result = freeArray[--countFree];
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
      return(new MYY10121_IA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ImpIyy1ParentPinstanceId_AS = ' ';
      ImpIyy1ParentPinstanceId = "00000000000000000000";
      ImpIyy1ParentPkeyAttrText_AS = ' ';
      ImpIyy1ParentPkeyAttrText = "     ";
    }
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IImportView orig )
    {
      this.CopyFrom((MYY10121_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( MYY10121_IA orig )
    {
      ImpIyy1ParentPinstanceId_AS = orig.ImpIyy1ParentPinstanceId_AS;
      ImpIyy1ParentPinstanceId = orig.ImpIyy1ParentPinstanceId;
      ImpIyy1ParentPkeyAttrText_AS = orig.ImpIyy1ParentPkeyAttrText_AS;
      ImpIyy1ParentPkeyAttrText = orig.ImpIyy1ParentPkeyAttrText;
    }
  }
}
